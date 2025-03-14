import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteriaList
from .helper import StopOnTokens
import deepspeed

from torch import LongTensor, FloatTensor

import logging
logger = logging.getLogger('RRAG-main')

import litellm
from litellm import batch_completion
from openai import OpenAI
import os 
import json
import joblib
from .prompt_template import *
MAX_NEW_TOKENS = 20
CONTEXT_MAX_TOKENS = {'/scratch/gpfs/zs7353/Mistral-7B-Instruct-v0.2': 8192, 
                      '/scratch/gpfs/zs7353/Llama-3.2-3B-Instruct': 4096, 
                      '/scratch/gpfs/zs7353/Llama-3.1-8B-Instruct': 8192, 
                      '/scratch/gpfs/zs7353/Mixtral-8x7B-Instruct-v0.1': 32000,
                      '/scratch/gpfs/zs7353/DeepSeek-R1-Distill-Qwen-7B': 8192,
                      'gpt-4o':8192,
                      'o1-mini':8192,
                      '/scratch/gpfs/zs7353/vicuna-7b-v1.5': 4096}


def create_model(model_name,**kwargs):
    if model_name == 'mistral7b':
        return HFModel('/scratch/gpfs/zs7353/Mistral-7B-Instruct-v0.2',MISTRAL_TMPL,**kwargs) 
    elif model_name == 'deepseek7b':
        return HFModel('/scratch/gpfs/zs7353/DeepSeek-R1-Distill-Qwen-7B',DEEPSEEK_TMPL,**kwargs)
    elif model_name == 'llama3b':
        return HFModel('/scratch/gpfs/zs7353/Llama-3.2-3B-Instruct',LLAMA_TMPL,**kwargs) 
    elif model_name == 'gpt-4o':
        return GPTModel('gpt-4o', GPT_TMPL, **kwargs) 
    elif model_name == 'llama8b':
        return HFModel('/scratch/gpfs/zs7353/Llama-3.1-8B-Instruct',LLAMA_TMPL,**kwargs) 
    elif model_name == 'vicuna7b':
        return HFModel('/scratch/gpfs/zs7353/vicuna-7b-v1.5',VICUNA_TMPL,**kwargs)  
    elif model_name == 'mixtral8x7b':
        return HFModel('/scratch/gpfs/zs7353/Mixtral-8x7B-Instruct-v0.1',MISTRAL_TMPL,**kwargs) 
    elif model_name == 'o1-mini':
        return GPTModel('o1-mini', GPT_TMPL, **kwargs) 
    else:
        raise NotImplementedError

class BaseModel:
    def __init__(self,cache_path=None):
        # setup the LLM response cache if cache_path is not None
        self.use_cache = cache_path is not None 
        self.cache_path = cache_path
        if cache_path is not None and os.path.exists(cache_path):
            self.cache = self.load_cache()
        else:
            self.cache = {}
        self.hash = lambda x: x # for now directly string as the hash key
        # cache is a dict {hash(s):LLM response for input s}
        # 
        # end of sentence str
        self.clean_str = []

        # input prompt template
        self.prompt_template = {}

    def query(self, prompt):
        if self.use_cache: # use cache if cache hits
            result = self.query_from_cache(prompt)
            if len(result)>0:
                return result

        # otherwise, do normal LLM query
        result = self._query(prompt)

        # store the response to self.cache
        if self.use_cache:
            self.cache[self.hash(prompt)]=result
        return result

    def _query(self,prompt): # will be implemented in each subclass
        raise NotImplementedError

    def batch_query(self, prompt_list):
        if self.use_cache: # use cache if cache hits
            results = self.batch_query_from_cache(prompt_list)
            if len(results)>0:
                return results

        # otherwise, do normal LLM batch query

        results = self._batch_query(prompt_list)

        # store the response to self.cache
        if self.use_cache:
            for p,r in zip(prompt_list,results):
                self.cache[self.hash(p)]=r
        return results

    def _batch_query(self, prompt_list):  # will be implemented in each subclass
        raise NotImplementedError

    def query_from_cache(self,prompt):
        # return cached responses
        h = self.hash(prompt)
        if h in self.cache:
            return self.cache[h]
        else:
            return ''

    def batch_query_from_cache(self,prompt_list):
        # return cached responses
        h_list = [self.hash(prompt) for prompt in prompt_list]
        if all([h in self.cache for h in h_list]):
            return [self.cache[h] for h in h_list]
        else:
            return []


    def dump_cache(self): # dump cache to disk
        joblib.dump(self.cache,self.cache_path)
    
    def load_cache(self): # load cache from disk
        return joblib.load(self.cache_path)
        
    def _clean_response(self,response): # clean response based on self.clean_str
        for pattern in self.clean_str:
            idx = response.find(pattern)
            if idx!=-1:
                response = response[:idx]
        return response.strip()

    def query_biogen(self, prompt):
        raise NotImplementedError

    def wrap_prompt(self,data_item,as_multi_choice=True,hints=None,seperate=False):  
        # use data_item and generate the input prompt to the LLM

        # data_item should be the output of DataUtils.process_data_item()
        # as_multi_choice: if use it as a multiple-choice QA
        # hints: if we are using hints in the last step of the keyword aggregation
        # seperate: if True, return a list of prompts for differnet passages; otherwise, concatenate all passages and return a single prompt

        # get info
        question = data_item['question']
        topk_content = data_item['topk_content']
        choices = data_item.get('choices',[])
        use_retrieval = len(topk_content) > 0 # if we use retrieved passage

        def fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints):
            filling = {'query_str': question}
            if use_retrieval:
                filling.update({'context_str': context_str})
            if as_multi_choice:
                filling.update({
                        'A': choices[0],
                        'B': choices[1],
                        'C': choices[2],
                        'D': choices[3]                
                    })
            if hints is not None:
                filling.update({'hints':hints})
            return template.format(**filling)

        # get corresponding template
        mode = 'qa'
        if as_multi_choice: mode += '-mc'
        if "long_gen" in data_item: mode += '-long'
        if not use_retrieval: mode += '-zero'
        if "decode" in data_item: mode += '-decode' 
        if "genhint" in data_item: mode += '-genhint'
        if hints is not None: mode += '-hint' 
        template = self.prompt_template[mode]

        if seperate:
            return [fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints) for context_str in topk_content]
        else:
            context_str = '\n\n'.join(topk_content)
            return fill_template(template,question,context_str,choices,use_retrieval,as_multi_choice,hints)


class HFModel(BaseModel):
    def __init__(self, model_name, prompt_template, cache_path=None, max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        # set max number of output tokens
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens 

        # set up tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 
        if "CohereForAI" in model_name:
            #'CohereForCausalLM' object has no attribute 'torch_dtype'
            self.model = AutoModelForCausalLM.from_pretrained(model_name,**kwargs) 
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda',**kwargs)
        
        self.prompt_template = prompt_template
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name

        with open("configs/ds_config.json", 'r') as f:
            ds_config = json.load(f)

        # self.model = deepspeed.init_inference(
        #     self.model,
        #     config=ds_config,
        #     mp_size=4,    
        #     max_tokens=8192,
        #     replace_method="auto"
        # )

        self.generation_kwargs = {
            'max_new_tokens':self.max_output_tokens,
            'pad_token_id':self.tokenizer.eos_token_id,
            'do_sample':True
        }
        if 'Llama-3' in model_name:
            self.generation_kwargs['eos_token_id']=[self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        self.clean_str = ['\n\n'] 


    def _query(self, prompt):
        # get text prompt as input and return text responses
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        token_length = inputs.input_ids.size(1)
        # check if the num of token exceeds the max context window ..
        # this only happens when we use the bio generation data without any defense (i.e., we concatenate all long passages together)
        # we did not implement this cut off for _batch_query
        if token_length >= (CONTEXT_MAX_TOKENS[self.model_name]-self.max_output_tokens):
            prompt_length = len(prompt)
            ratio = (CONTEXT_MAX_TOKENS[self.model_name]-self.max_output_tokens-100)/token_length # this is a rough cut off, 100 is a buffer
            ## this is left cut 
            # prompt_cut = prompt[:int(prompt_length*ratio)] + "..." + "\n\n" + prompt[prompt.rfind("Query: Tell me a bio of"):] 
            ## this is right cut 
            prompt_cut = "Context information is below.\n" + "---------------------\n ..."+ prompt[-int(prompt_length*ratio):]
            inputs = self.tokenizer(prompt_cut, return_tensors="pt").to("cuda")
            logger.warning(f"Prompt length exceeds the limit, cut the prompt to {inputs.input_ids.size(1)} tokens")

        outputs = self.model.generate(**inputs,**self.generation_kwargs)
        outputs = outputs[0][len(inputs[0]):]
        result = self.tokenizer.decode(outputs, skip_special_tokens=True)
        result = self._clean_response(result)
        return result
    

    def _batch_query(self, prompt_list):
        # get a list of text prompts as input and return a list text responses
        inputs = self.tokenizer(prompt_list, return_tensors="pt",
                                    padding=True, 
                                    #padding='max_length',
                                    truncation=True,
                                    #max_length=800
                                    ).to("cuda")
        outputs = self.model.generate(**inputs, **self.generation_kwargs)
        results = self.tokenizer.batch_decode(outputs[:, len(inputs[0]):],skip_special_tokens=True)
        results = [self._clean_response(x) for x in results]
        
        return results


class GPTModel(BaseModel):
    def __init__(self, model_name, prompt_template, cache_path=None,max_output_tokens=None, **kwargs):
        super().__init__(cache_path)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = 1.0
        self.max_output_tokens = MAX_NEW_TOKENS if max_output_tokens is None else max_output_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    def _query(self, prompt): 
        try:
            if self.model_name == "o3-mini" or self.model_name == "o1-mini" or self.model_name == "o1":
                chat = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=self.max_output_tokens
                )   
            else:
                chat = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens
                )
            response = chat.choices[0].message.content
        except Exception as e:
            print(e)
            response = ""
        return response
        

    def _batch_query(self,prompt_list):

        prompt_list_with_template = []
        for prompt in prompt_list:
            prompt_list_with_template.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ])
        result = batch_completion(model=self.model_name, messages=prompt_list_with_template, max_tokens=self.max_output_tokens)
        response_list = [x.choices[0].message.content for x in result]

        return response_list

    def query_biogen(self, prompt):
        return self.query(prompt)
