import logging

from collections import Counter,defaultdict

from transformers import StoppingCriteriaList, MaxLengthCriteria, AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
stopword_set = set(stopwords.words('english'))

import torch 
from itertools import combinations
from .helper import clean_str, StopOnTokens
import copy
from tqdm import tqdm
from torch import LongTensor, FloatTensor
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import spacy
import os
import json
import random
from transformers import pipeline
from itertools import chain, combinations
import math

from src.decoding_methods import secure_decoding

logger = logging.getLogger('RRAG-main')

INJECTION = True # injection attack. if False, we consider passage modification attacks discussed in the appendix

def save_all_responses(save_path,response_list,data_item):
    all_data = []# it is a bit ugly... unnecessary read and write ; TODO: change it to jsonl instead
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        with open(save_path,'r') as f:
            all_data = json.load(f)
    all_data.append({"query":data_item['question'],
                     "answer":data_item['answer'],
                     "response":response_list})
    with open(save_path,'w') as f:
        json.dump(all_data,f,indent=4)


class RRAG:

    def __init__(self,llm):
        self.llm = llm

    def query_undefended(self,data_item):
        query_prompt = self.llm.wrap_prompt(data_item,as_multi_choice='choices' in data_item)
        #response = None 
        response =  self.llm.query(query_prompt)
        logger.debug(f'Query_prompt:\n{query_prompt}')
        logger.debug(f'Response:\n{response}')
        logger.debug(f'Answer:\n{data_item["answer"]}')
        return response

    def query(self, data_item):
        raise NotImplementedError

    def _eval_response(self,response,data_item):
        answer = data_item['answer']
        response = clean_str(response)
        for ans in answer:
            if clean_str(ans) in response:
                return True 
        return False


class GraphBasedRRAG(RRAG):

    def __init__(self,llm):
        self.llm = llm
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu" # for gpt-4o
        self.nli_tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/zs7353/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("/scratch/gpfs/zs7353/DeBERTa-v3-large-mnli-fever-anli-ling-wanli").to(device)

    def query(self, data_item):
        docs = data_item['topk_content']
        seperate_responses = self.llm.batch_query(self.llm.wrap_prompt(data_item,as_multi_choice=False,seperate=True))
        k = len(docs)
        # Build pairwise prompts to check for contradictory information
        prompts = []
        out_edges = {i: set() for i in range(k)}
        in_edges = {i: set() for i in range(k)}

        premises = []
        hypotheses = []
        pair_indices = []

        for i in range(k):
            for j in range(i + 1, k):
                premise = f"The answer to the question: {data_item['question']}\nis {seperate_responses[i]}."
                hypothesis = f"The answer to the question: {data_item['question']}\nis {seperate_responses[j]}."
                premises.append(premise)
                hypotheses.append(hypothesis)
                pair_indices.append((i, j))

        if premises:
            inputs = self.nli_tokenizer(premises, hypotheses, return_tensors='pt', truncation=True, padding=True)
            inputs = {key: value.to(self.nli_model.device) for key, value in inputs.items()}

            # Run the model on the batch
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

            # Process each batch item and update edges based on contradiction probability
            for idx, (i, j) in enumerate(pair_indices):
                contradiction_probability = probs[idx][2].item()
                if contradiction_probability >= 0.5 and "I don't know" not in seperate_responses[i] and "I don't know" not in seperate_responses[j]:
                    out_edges[i].add(j)
                    in_edges[j].add(i)
        
        # Iteratively remove vertices with out-degree greater than (number of remaining vertices)/2
        # remaining = set(range(k))
        remaining = set()
        # just don't take irrelevant docs? They are just noisy and useless
        for i in range(k):
            if "I don't know" not in seperate_responses[i]:
                remaining.add(i)
        
        removal_occurred = True
        while removal_occurred:
            removal_occurred = False
            current_remaining = list(remaining)
            n_remaining = len(remaining)
            to_remove = []
            for v in current_remaining:
                current_out_degree = len(out_edges[v].intersection(remaining))
                if current_out_degree > math.floor(n_remaining / 2):
                    to_remove.append(v)
            if to_remove:
                removal_occurred = True
                for v in to_remove:
                    remaining.discard(v)
        
        # From the remaining documents, select those with in-degree 0
        selected = []
        for v in remaining:
            current_in_degree = len(in_edges[v].intersection(remaining))
            if current_in_degree == 0:
                selected.append(v)
        
        logger.info(selected)
        # Fallback: if no document has in-degree 0, use all remaining documents
        if not selected:
            selected = list(remaining)
        
        # Sort selected documents by their original rank order
        selected.sort()
        
        # Update the data_item to include only the selected documents
        new_data_item = data_item.copy()
        new_data_item['topk_content'] = [docs[i] for i in selected]
        
        # Create the final prompt using the LLM's wrap_prompt method
        ultimate_prompt = self.llm.wrap_prompt(new_data_item, as_multi_choice=False, seperate=False)
        
        # Return the final answer by querying the LLM
        final_answer = self.llm._query(ultimate_prompt)
        print("final_answer: ", final_answer)
        return final_answer



class MISBasedRRAG(RRAG):

    def __init__(self, llm):
        self.llm = llm
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"  # for gpt-4o
        self.nli_tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/zs7353/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("/scratch/gpfs/zs7353/DeBERTa-v3-large-mnli-fever-anli-ling-wanli").to(device)

    def query(self, data_item):
        # Retrieve the documents and get separate responses.
        docs = data_item['topk_content']
        seperate_responses = self.llm.batch_query(self.llm.wrap_prompt(data_item, as_multi_choice=False, seperate=True))
        k = len(docs)
        
        # Build an undirected graph: graph[i] holds all vertices j that contradict with document i.
        graph = {i: set() for i in range(k)}
        premises = []
        hypotheses = []
        pair_indices = []

        for i in range(k):
            for j in range(i + 1, k):
                premise = f"The answer to the question: {data_item['question']}\nis {seperate_responses[i]}."
                hypothesis = f"The answer to the question: {data_item['question']}\nis {seperate_responses[j]}."
                # premise = docs[i]
                # hypothesis = docs[j]
                premises.append(premise)
                hypotheses.append(hypothesis)
                pair_indices.append((i, j))
        
        if premises:
            inputs = self.nli_tokenizer(premises, hypotheses, return_tensors='pt', truncation=True, padding=True)
            inputs = {key: value.to(self.nli_model.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
            # For each pair, add an undirected edge if the answers contradict.
            for idx, (i, j) in enumerate(pair_indices):
                contradiction_probability = probs[idx][2].item()
                if (contradiction_probability >= 0.5 and "I don't know" not in seperate_responses[i] and "I don't know" not in seperate_responses[j]):
                # logger.info(docs[i])
                # logger.info(docs[j])
                # logger.info(contradiction_probability)
                # if contradiction_probability >= 0.5:
                    graph[i].add(j)
                    graph[j].add(i)
        
        z = {i for i in range(k) if "I don't know" not in seperate_responses[i]}
        
        # Compute the maximum independent set over the vertices.
        # Among all maximum independent sets, choose the one with the lexicographically smallest order.
        best_set = self._max_independent_set(graph, z)
        
        # Fallback: if best_set is empty, use all z documents.
        if not best_set:
            best_set = list(z)
            if not best_set:
                best_set = [i for i in range(k)]
        else:
            best_set = list(best_set)

        best_set.sort()  # sort in ascending order (better ranked docs have lower indices)
        logger.info(f"Selected document indices: {best_set}")
        
        # Update data_item with only the selected documents.
        new_data_item = data_item.copy()
        new_data_item['topk_content'] = [docs[i] for i in best_set]
        
        # Create the final prompt and query for the ultimate answer.
        ultimate_prompt = self.llm.wrap_prompt(new_data_item, as_multi_choice=False, seperate=False)
        final_answer = self.llm._query(ultimate_prompt)
        print("final_answer:", final_answer)
        return final_answer

    def _max_independent_set(self, graph, vertices):
        best_size = 0
        best_sets = []
        vertices_list = list(vertices)
        
        # Generate all subsets of vertices_list
        for subset in chain.from_iterable(combinations(vertices_list, r) for r in range(len(vertices_list) + 1)):
            subset = set(subset)
            if self._is_independent(subset, graph):
                subset_size = len(subset)
                if subset_size > best_size:
                    best_size = subset_size
                    best_sets = [tuple(sorted(subset))]
                elif subset_size == best_size:
                    best_sets.append(tuple(sorted(subset)))
                    
        # Return the lexicographically smallest independent set (as a tuple).
        if best_sets:
            return min(best_sets)
        else:
            return set()

    def _is_independent(self, subset, graph):
        for v in subset:
            for u in subset:
                if u != v and u in graph[v]:
                    return False
        return True


class WeightedMajorityVoting(RRAG):
    def __init__(self, llm, gamma=1):
        self.llm = llm
        self.gamma = gamma

    def query(self, data_item):
        # assume the prompt ask the LLM to output A., B., C., D., or E. No information found
        seperate_responses = self.llm.batch_query(self.llm.wrap_prompt(data_item,as_multi_choice=True,seperate=True))
        seperate_preds = []
        for response in seperate_responses:
            if "gpt" in self.llm.model_name: 
                if response.find('Answer') != -1:
                    response = response[(response.find('Answer')+7):].strip()
                else:
                    response = response.strip()
                if response[0] in 'ABCD':
                    seperate_preds.append(response[0]+'.')
                else:
                    seperate_preds.append('E.')
            else:
                response = response.strip()
                if len(response)>=2 and response[1]=='.' and response[0] in'ABCD':
                    seperate_preds.append(response[:2])
                else:
                    seperate_preds.append('E.')

        logger.debug(f'Seperate responses: {seperate_preds}')

        cntr = defaultdict(float)

        total_weight = 0
        total_weight_orig = 0
        for i, pred in enumerate(seperate_preds):
            if pred == 'E.':
                continue
            weight = self.gamma ** i  # First position weight=1, second=gamma, third=gamma^2, etc.
            total_weight += weight
            total_weight_orig += 1

        for i, pred in enumerate(seperate_preds):
            if pred == 'E.':
                continue 
            weight = self.gamma ** i      
            cntr[pred] += weight * total_weight_orig / total_weight
        
        cntr = Counter(cntr)
        cntr = cntr.most_common(2)

        if len(cntr)==0:
            pred = 'E.' # No information found.
        else:
            pred = cntr[0][0] 
        return pred


class WeightedKeywordAgg(RRAG):

    def __init__(self,llm,relative_threshold=0.3, absolute_threshold=3, abstention_threshold=1, gamma=1, longgen=False):
        self.llm = llm
        self.abstention_threshold = 1
        self.keyword_extractor = spacy.load("en_core_web_sm") 
        self.ignore_set = {'VERB','INTJ','ADP','AUX','CCONJ','DET','PART','PRON','SCONJ','PUNCT','SPACE'}
        self.absolute = absolute_threshold
        self.relative = relative_threshold
        self.gamma = gamma
        self.longgen = longgen # if it is long-form generation or short-form (we use slightly different prompt template)
        logger.info(f'abs: {absolute_threshold}, relative: {relative_threshold}')

    def query(self, data_item, abstention_threshold=None): 
        # override original threshold parameters if given
        abstention_threshold = abstention_threshold if abstention_threshold is not None else self.abstention_threshold
        if self.longgen:
            data_item['genhint'] = True # add a flag so that wrap_prompt() can retrieve the correct prompt template
        # make seperate predictions
        seperate_responses_raw = self.llm.batch_query(self.llm.wrap_prompt(data_item,as_multi_choice=False,seperate=True))
        abstained_idx = []
        seperate_responses = []
        logger.debug(f'Seperate responses:\n')
        total_weight = 0
        total_weight_orig = 0
        for i,x in enumerate(seperate_responses_raw):
            logger.debug(f'{i}: {x}\n')
            if "I don't" in x:
                abstained_idx.append(i)
            else:
                seperate_responses.append((x,  self.gamma ** i))
                total_weight +=  self.gamma ** i
                total_weight_orig += 1

        logger.debug(f'Number of retained responses: {len(seperate_responses)}')

        if len(seperate_responses) < abstention_threshold:
            logger.warning('Abstain from making response...')
            return "I don't know."
        
        def construct_phrase(token_list):
            ret = ''
            for token in token_list:
                ret+=token.lemma_+token.whitespace_
        # extract keyword/keyphrase
        all_extracted_phrase = []
        token_counter = defaultdict(int)
        for response, weight in seperate_responses:
            doc = self.keyword_extractor(response)
            phrase_list = [response.strip()] 
            tmp = []
            for token in doc:
                if token.pos_ in self.ignore_set:
                    if len(tmp)>0:
                        phrase = ''.join([x.lemma_+x.whitespace_ for x in tmp]).strip()
                        phrase_list.append(phrase)
                        phrase_list+=[x.lemma_ for x in tmp]
                        tmp = []
                else:
                    tmp.append(token)

            phrase = ''.join([x.lemma_+x.whitespace_ for x in tmp]).strip()
            phrase_list.append(phrase)
            phrase_list+=[x.lemma_ for x in tmp]
            phrase_list = set(phrase_list) # only consider unique keywords
            all_extracted_phrase.append(phrase_list)
            for phrase in phrase_list:
                token_counter[phrase]+=weight * total_weight_orig / total_weight

        # filtering 
        print(phrase_list)
        count_threshold = min(self.absolute,self.relative*len(seperate_responses))
        logger.debug(sorted(token_counter.items(), key=lambda x: (len(x[0]),x[0]), reverse=True))
        logger.debug(f'count_threshold,{count_threshold}')
        for token,count in list(token_counter.items()):
            if (count < count_threshold) or (token in punctuation) or (token in stopword_set)  or (self.longgen and ' ' not in token): # if it is long generation, we remove single words to reduce the size the keyword set...
                del token_counter[token]

        # generate keyword hints
        sorted_tokens = sorted(token_counter.items(), key=lambda x: (len(x[0]),x[0]), reverse=True)
        hints = ', '.join([f'{token}' for token,count in sorted_tokens])
        logger.debug(sorted_tokens)
        query_prompt = self.llm.wrap_prompt(data_item,as_multi_choice=False,hints=hints)
        logger.debug(f'Hint prompt:\n{query_prompt}')
        response = self.llm.query(query_prompt)
        logger.debug(f'Keyword aggregated response:\n{response}')

        return response


class WeightedDecodingAgg(RRAG):
    def __init__(self,llm, eta, gamma=1, abstention_prob=None):
        self.llm = llm
        self.llm.model.secure_decoding = secure_decoding.__get__(self.llm.model, type(self.llm.model))
        self.temperature = 1.0 #args.temperature
        abstention_prob_list = {'/scratch/gpfs/zs7353/Llama-3.2-3B-Instruct': 0.99, 
                                '/scratch/gpfs/zs7353/Mistral-7B-Instruct-v0.2': 0.99, 
                                '/scratch/gpfs/zs7353/DeepSeek-R1-Distill-Qwen-7B': 0.99}
        if abstention_prob is None:
            self.abstention_prob = abstention_prob_list.get(llm.model_name, 0.99)
            logger.debug(f"Using default abstention probability: {self.abstention_prob}")

        self.gamma = gamma
        self.eta = eta
       
    def preprocess_input(self,data_item):
        prompt_list = self.llm.wrap_prompt(data_item,as_multi_choice=False,seperate=True)
        data_item_zero_shot = {"question": data_item["question"], "topk_content":[], "long_gen": True}
        prompt_zero_shot = self.llm.wrap_prompt(data_item_zero_shot,as_multi_choice=False,seperate=False)
        prompt_list.append(prompt_zero_shot)

        prompt_list_draft = [prompt + " I don't know" for prompt in prompt_list]

        # batched version 
        input_dict_draft = self.llm.tokenizer(prompt_list_draft, return_tensors="pt", padding=True).to("cuda")
        input_ids_draft = input_dict_draft.input_ids.to("cuda")
        attention_mask_draft = input_dict_draft.attention_mask.to("cuda")

        # compute the perplexity of the prompt "I don't know"
        with torch.no_grad():
            output_token_draft = self.llm.model(input_ids_draft, attention_mask=attention_mask_draft)
            logits_draft = output_token_draft.logits

        probs = torch.softmax(logits_draft, dim=-1)
        total_probability = torch.ones(input_ids_draft.shape[0]).to("cuda")

        input_dict = self.llm.tokenizer(prompt_list, return_tensors="pt", padding= True)
        start_index = input_dict.input_ids.size(1)

        for i in range(start_index, input_ids_draft.size(1) - 1):  # Exclude the last token since there's no next token to predict
            # Get the probability of the actual next token
            next_token_id = input_ids_draft[0, i + 1]  # The next token in the sequence
            next_token_prob = probs[:, i, next_token_id]
            total_probability *= next_token_prob

        #print(f"total_probability: {total_probability}")
        input_ids = input_dict.input_ids.to("cuda")
        attention_mask = input_dict.attention_mask.to("cuda")

        # filter the prompt with the probability of "I don't know" is greater than 0.9
        total_probability[-1] = 0.0 # last one is the zero-shot prompt
        ab_record = total_probability < self.abstention_prob
        input_ids = input_ids[ab_record]
        attention_mask = attention_mask[ab_record]
        return input_ids,attention_mask,ab_record

    def query(self, data_item):

        input_ids,attention_mask,ab_record = self.preprocess_input(data_item)

        if input_ids.shape[0] == 1: # only the no-retrieval prediction
            return "I don't know.", False
        
        # Initialize past_key_values for caching
        past_key_values = None
        generated_outputs = []

        stop_list = ["\n#", "\n##","\n###","\n####","\n#####"] + ["\n\n"] ################ seems to work fine
        stop_token_ids = [self.llm.tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
        stop_token_ids = [LongTensor(x).to("cuda") for x in stop_token_ids]
        stopping_criteria = StoppingCriteriaList([
            MaxLengthCriteria(max_length=len(input_ids[0]) + self.llm.max_output_tokens),
            StopOnTokens(stop_token_ids=stop_token_ids)
        ])
        
        generated_outputs = self.llm.model.secure_decoding(input_ids,
                                                           attention_mask=attention_mask,
                                                           stopping_criteria=stopping_criteria,
                                                           use_cache=False,
                                                           pad_token_id=self.llm.tokenizer.pad_token_id,
                                                           eos_token_id=self.llm.tokenizer.eos_token_id,
                                                           return_dict_in_generate=True,
                                                           temperature=self.temperature,
                                                           tokenizer=self.llm.tokenizer,
                                                           eta=self.eta,
                                                           gamma=self.gamma)

        generated_output_text = self.llm.tokenizer.decode(generated_outputs, skip_special_tokens=True)
        return generated_output_text
    

class RandomSamplingReQueryAgg(RRAG):
    def __init__(
        self, 
        llm,
        sample_size=5, 
        num_samples=3,
        gamma=1
    ):
        super().__init__(llm)
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.gamma = gamma

        self.use_openai = False
        self.openai_model = "text-embedding-ada-002"
        self.hf_model_name = "/scratch/gpfs/bi0600/all-mpnet-base-v2"

        if not self.use_openai:
            self.hf_model = SentenceTransformer(self.hf_model_name)
        else:
            self.client = OpenAI()

    def get_openai_embeddings(self, text_list):
        response = self.client.embeddings.create(
            model=self.openai_model,
            input=text_list
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def get_hf_embeddings(self, text_list):
        embeddings = self.hf_model.encode(text_list)
        return embeddings

    def query(self, data_item):
        question = data_item["question"]
        all_chunks = data_item["topk_content"]
        n = len(all_chunks)

        # 1) Assign geometric weights to chunks: gamma^i
        weights = np.array([self.gamma ** i for i in range(n)])
        weights /= weights.sum()  # normalize

        # 2) First-stage sampling: sample multiple subsets & query LLM
        sampled_responses = []
        for i in range(self.num_samples):
            sampled_chunks = list(
                np.random.choice(
                    all_chunks,
                    size=min(self.sample_size, n),
                    replace=False,
                    p=weights
                )
            )
            prompt = self.build_prompt(question, sampled_chunks)
            response = self.llm.query(prompt)
            sampled_responses.append(response)

        logger.debug(f"First-stage sampled responses:\n{sampled_responses}")

        # 3) Second-stage: pick the response closest to the mean embedding
        if self.use_openai:
            response_embeddings = self.get_openai_embeddings(sampled_responses)
        else:
            response_embeddings = self.get_hf_embeddings(sampled_responses)

        # Compute average (centroid) embedding
        response_embeddings = np.array(response_embeddings)
        avg_embedding = np.mean(response_embeddings, axis=0)

        # Find whichever response is closest to this centroid
        best_idx = None
        best_sim = -float("inf")
        for i, emb in enumerate(response_embeddings):
            cos_sim = dot(emb, avg_embedding) / (norm(emb) * norm(avg_embedding))
            if cos_sim > best_sim:
                best_idx = i
                best_sim = cos_sim
        
        final_response = sampled_responses[best_idx]
        logger.debug(f"Second-stage final response:\n{final_response}")
        return final_response

    def build_prompt(self, question, chunks):
        context_text = "\n\n".join(chunks)
        return f"Answer the following question based on the context below. It is very important that the answer should be based solely on evidence found in the context information. The answer should be as short as possible and can only use words found in the context information. \n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"



class SamplingWithKeyWordAggregation(RRAG):
    def __init__(
        self, 
        llm,
        sample_size=5, 
        num_samples=3,
        gamma=1,
        relative_threshold=0.3,
        absolute_threshold=3,
        abstention_threshold=1,
    ):
        super().__init__(llm)
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.gamma = gamma

        self.keyword_extractor = spacy.load("en_core_web_sm") 
        self.ignore_set = {'VERB','INTJ','ADP','AUX','CCONJ','DET','PART','PRON','SCONJ','PUNCT','SPACE'}

        self.abstention_threshold = abstention_threshold
        self.absolute = absolute_threshold
        self.relative = relative_threshold
        self.gamma = gamma
        logger.debug(f'Sampling+keyword. abs: {absolute_threshold}, relative: {relative_threshold}')


    def query(self, data_item):
        question = data_item["question"]
        all_chunks = data_item["topk_content"]
        n = len(all_chunks)

        # 1) Assign geometric weights to chunks: gamma^i
        weights = np.array([self.gamma ** i for i in range(n)])
        weights /= weights.sum()  # normalize

        # 2) First-stage sampling: sample multiple subsets & query LLM
        sampled_responses = []

        for i in range(self.num_samples):
            indices = np.random.choice(
                n,
                size=min(self.sample_size, n),
                replace=False,
                p=weights
            )
            sampled_chunks = [all_chunks[j] for j in indices]
            prompt = self.build_prompt(question, sampled_chunks)
            response = self.llm.query(prompt)

            if "I don't" not in response:
                sampled_responses.append(response)

        logger.debug(f"Sampled responses:\n{sampled_responses}")

        if len(sampled_responses) < self.abstention_threshold:
            logger.warning("Abstain from making response...")
            return "I don't know."

        # 3) Keyword aggregation
        token_counter = defaultdict(int)
        all_extracted_phrase = []

        for response in sampled_responses:
            doc = self.keyword_extractor(response)
            phrase_list = [response.strip()]
            tmp = []

            for token in doc:
                if token.pos_ in self.ignore_set:
                    if len(tmp) > 0:
                        phrase = ''.join([x.lemma_ + x.whitespace_ for x in tmp]).strip()
                        phrase_list.append(phrase)
                        phrase_list += [x.lemma_ for x in tmp]
                        tmp = []
                else:
                    tmp.append(token)

            phrase = ''.join([x.lemma_ + x.whitespace_ for x in tmp]).strip()
            phrase_list.append(phrase)
            phrase_list += [x.lemma_ for x in tmp]
            phrase_list = set(phrase_list)

            all_extracted_phrase.append(phrase_list)
            for phrase in phrase_list:
                token_counter[phrase] += 1

        # Filtering
        print(phrase_list)
        count_threshold = min(self.absolute, self.relative * len(sampled_responses))
        for token, count in list(token_counter.items()):
            if (count < count_threshold) or (token in punctuation) or (token in stopword_set):
                del token_counter[token]

        # Generate keyword-based final query
        sorted_tokens = sorted(token_counter.items(), key=lambda x: (len(x[0]), x[0]), reverse=True)
        hints = ', '.join([token for token, _ in sorted_tokens])
        logger.debug("Sorted tokens for hints:")
        logger.debug(sorted_tokens)
        hint_prompt = self.llm.wrap_prompt(data_item, as_multi_choice=False, hints=hints)
        logger.debug(f'Hint prompt:\n{hint_prompt}')
        final_response = self.llm.query(hint_prompt)

        logger.debug(f"Final response:\n{final_response}")
        return final_response

    def build_prompt(self, question, chunks):
        context_text = "\n\n".join(chunks)
        return f"""
        Given the context information below and not prior knowledge, answer the query with only keywords.
        If there is no relevant information, just say "I don't know".\n\n
        Context:\n
        {context_text}\n\n
        Query: {question}\n
        Answer:
        """
