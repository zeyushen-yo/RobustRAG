import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers.utils import ModelOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.utils import _crop_past_key_values


@torch.no_grad()
def secure_decoding(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        temperature: Optional[float] = 0.01,
        tokenizer: Optional = None,
        eta: Optional[float] = 1.0,
        gamma: Optional[float] = 1.0,
        **model_kwargs,
):
    # init values
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

    # # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None

    max_len = stopping_criteria[0].max_length

    generated_tokens = []
    ii = 0 
    while True:
        ii+=1  
        with torch.no_grad():
            outputs = self(input_ids, **model_kwargs)

        new_logits_full = outputs.logits[:, -1, :]  # last token prediction
        no_retrieval_token = new_logits_full[-1,:].argmax()
        new_logits = new_logits_full[:-1,:] # remove no-retrieval prompt

        # get the softmax of the next_token_logits
        next_token_confidence = torch.nn.functional.softmax(new_logits/temperature, dim=-1)
        kk = next_token_confidence.shape[0]
        weights = gamma ** torch.arange(kk, device=next_token_confidence.device, dtype=next_token_confidence.dtype)
        weighted_confidence = next_token_confidence * weights.unsqueeze(1)
        next_token_agg = torch.sum(weighted_confidence, dim=0)
            
        if ii == 1:
            ## Here, we should reduce the probability of the "I" token
            ## We can do this by setting the probability of the "I" token to 0
            I_token = tokenizer("I", return_tensors="pt")["input_ids"][0][1].item()
            n_token = tokenizer("/n", return_tensors="pt")["input_ids"][0][1].item()
            Please_token = tokenizer("Please", return_tensors="pt")["input_ids"][0][1].item()
            next_token_agg[I_token] = 0  # set the probability of the "I" token to 0
            next_token_agg[n_token] = 0 # set the probability of the "/n" token to 0
            next_token_agg[Please_token] = 0 # set the probability of the "Please" token to 0

        top2_tokens = torch.topk(next_token_agg, 2)
        if top2_tokens.values[0] - top2_tokens.values[1]>eta:
            selected_tokens = top2_tokens.indices[0] # top 1 token
        else: # no retrieval token
            selected_tokens = no_retrieval_token

        input_ids = torch.cat((input_ids, selected_tokens.repeat(input_ids.shape[0], 1)), dim=-1)
        new_cur_len = input_ids.shape[-1]
        new_cache_size = new_cur_len - 1
        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)
        model_kwargs["past_key_values"] = outputs.past_key_values
        model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], torch.ones_like(model_kwargs["attention_mask"][:, :1])], dim=-1)
        generated_tokens.append(selected_tokens)

        # stop if we exceed the maximum length
        if (selected_tokens == eos_token_id_tensor.item()).any():
            break
        
        if all(stopping_criteria(input_ids, scores)):
            break

    return generated_tokens


def greedy_rag_decoding(model, tokenizer, stopping_criteria, query_ids, query_attention_mask, doc_ids_list, 
                        uncertain_thres=0.1, malicious_thres=0.2):
    included_docs = []        # list of currently included document id tensors
    flagged_docs = set()      # indices of documents flagged as malicious
    # Start with the most trustworthy document (assuming the docs are ranked in terms of trustworthiness)
    if len(doc_ids_list) > 0:
        included_docs = [doc_ids_list[0]]

    # Helper to build concatenated input ids and attention mask from query and docs.
    def build_input(query_ids, query_mask, docs):
        input_ids = query_ids
        attention_mask = query_mask
        for doc_tensor in docs:
            input_ids = torch.cat([input_ids, doc_tensor], dim=1)
            doc_mask = torch.ones((1, doc_tensor.size(1)), dtype=query_mask.dtype, device="cuda")
            attention_mask = torch.cat([attention_mask, doc_mask], dim=1)
        return input_ids, attention_mask

    input_ids, attention_mask = build_input(query_ids, query_attention_mask, included_docs)
    generated_ids = []  # store generated token ids

    while True:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)
        next_token_logits = logits[:, -1, :]  # logits for next token
        probs = F.softmax(next_token_logits, dim=-1)[0]  # shape: (vocab_size,)
        top_probs, top_indices = torch.topk(probs, k=2)
        p_max = float(top_probs[0].item())
        p_second = float(top_probs[1].item()) if top_probs.size(0) > 1 else 0.0

        # If uncertainty (difference) less than uncertain_thres, try adding an unflagged doc.
        if p_max - p_second < uncertain_thres:
            for doc_idx in range(1, len(doc_ids_list)):
                if doc_idx in flagged_docs or any(doc_idx == incl_idx for incl_idx, _ in enumerate(included_docs)):
                    continue
                candidate_doc = doc_ids_list[doc_idx]
                temp_docs = included_docs + [candidate_doc]
                temp_input_ids, temp_mask = build_input(query_ids, query_attention_mask, temp_docs)
                temp_outputs = model(input_ids=temp_input_ids, attention_mask=temp_mask)
                temp_logits = temp_outputs.logits
                temp_next_logits = temp_logits[:, -1, :]
                temp_probs = F.softmax(temp_next_logits, dim=-1)[0]
                distribution_change = float(torch.sum(torch.abs(temp_probs - probs)).item() / 2.0)
                # setting a good malicious threshold is tricky..
                # if distribution_change > malicious_thres:
                #     flagged_docs.add(doc_idx)
                #     continue
                
                # Accept candidate document
                included_docs.append(candidate_doc)
                probs = temp_probs  # update distribution
                input_ids, attention_mask = temp_input_ids, temp_mask
                break

        # Recompute top token (using current probs)
        top_prob, top_index = torch.max(probs, dim=-1)
        next_token_id = top_index.item()
        generated_ids.append(next_token_id)

        print(tokenizer.eos_token_id, next_token_id)
        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
            break
        if all(stopping_criteria(torch.tensor([generated_ids], device="cuda"), None)):
            break
        
        new_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device="cuda")
        input_ids = torch.cat([input_ids, new_token_tensor], dim=1)
        new_mask = torch.ones((1, 1), dtype=attention_mask.dtype, device="cuda")
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)
        print(tokenizer.decode(generated_ids, skip_special_tokens=True))

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, list(flagged_docs)