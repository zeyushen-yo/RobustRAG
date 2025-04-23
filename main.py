import argparse
import os
import json
from tqdm import tqdm
import torch
import logging
from src.dataset_utils import load_data
from src.models import create_model
from src.defense import *
from src.baselines import *
from src.attack import *
from src.helper import get_log_name
import matplotlib.pyplot as plt
import pandas as pd
from llm_judge import LLMJudge
import time
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Robust RAG')

    # LLM settings
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed from distributed launcher")
    parser.add_argument('--model_name', type=str, default='mistral7b',choices=['mistral7b', 'llama3b', 'gpt-4o', 'o1-mini', 'deepseek7b', 'llama1b', 'tai_llama8b','tai_mistral7b'],help='model name')
    parser.add_argument('--dataset_name', type=str, default='realtimeqa',choices=['realtimeqa-mc','realtimeqa','open_nq','biogen', 'simpleqa'],help='dataset name')
    parser.add_argument('--model_dir', type=str, help='directory for huggingface models')
    parser.add_argument('--rep', type=int, default=1, help='number of times to repeat querying')
    parser.add_argument('--top_k', type=int, default=10,help='top k retrieval')

    # attack
    parser.add_argument('--attack_method', type=str, default='none',choices=['none','Poison','PIA'], help='The attack method to use (Poison or Prompt Injection)')
    parser.add_argument('--attackpos', type=int, default=0, help='The position of the attack in the top-k retrieval (0-indexed)')

    # defense
    parser.add_argument('--defense_method', type=str, default='keyword',choices=['none','voting','keyword','decoding', 'sampling', 'astuterag','instructrag_icl','graph','MIS', 'sampling_keyword',],help='The defense method to use')
    parser.add_argument('--alpha', type=float, default=0.3, help='keyword filtering threshold alpha')
    parser.add_argument('--beta', type=float, default=3.0, help='keyword filtering threshold beta')
    parser.add_argument('--eta', type=float, default=0.0, help='decoding confidence threshold eta')
    parser.add_argument('--T', type=int, default=10, help='number of samples for sampling method')
    parser.add_argument('--m', type=int, default=1, help='number of docs per sample for sampling method')
    parser.add_argument('--gamma', type=float, default=1.0, help='weight discount factor for reliability-aware methods (between 0 and 1)')

    # long gen certifcation # not really used in the paper
    parser.add_argument('--temperature', type=float, default=1.0, help='The temperature for softmax')

    # other
    parser.add_argument('--debug', action = 'store_true', help='output debugging logging information')
    parser.add_argument('--save_response', action = 'store_true', help='save the results for later analysis')
    parser.add_argument('--use_cache', action = 'store_true', help='save/use cache responses from LLM')
    parser.add_argument('--max_samples', type=int, default=None, help='limit maximum number of samples to run for testing') 
    parser.add_argument('--use_open_model_api', action = 'store_true', help='use APIs instead of local model for open models')

    args = parser.parse_args()
    return args

def main():
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    args = parse_args()
    LOG_NAME = get_log_name(args)
    logging_level = logging.DEBUG if args.debug else logging.INFO
    
    os.makedirs(f'log',exist_ok=True)
    
    logging.basicConfig(#level=logging_level,
        format=':::::::::::::: %(message)s'
    )

    logger = logging.getLogger('RRAG-main')
    logger.setLevel(level=logging_level)
    logger.addHandler(logging.FileHandler(f"log/{LOG_NAME}.log"))

    logger.info(args)

    data_tool = load_data(args.dataset_name,args.top_k)

    if args.use_cache: # use/save cached responses from LLM
        os.makedirs(f'cache/',exist_ok=True)
        cache_path = f'cache/{args.model_name}-{args.dataset_name}-{args.top_k}.z'
    else:
        cache_path = None

    if args.dataset_name == 'biogen':
        llm = create_model(args.model_name,args.model_dir, args.use_open_model_api, max_output_tokens=500)
        longgen = True
    else:
        llm = create_model(args.model_name,args.model_dir, args.use_open_model_api, cache_path=cache_path)
        longgen = False
 
    if args.defense_method == 'none': # no defense
        model = RRAG(llm)
    elif args.defense_method == 'voting': # weighted majority voting
        assert 'mc' in args.dataset_name
        model = WeightedMajorityVoting(llm)
    elif args.defense_method == 'keyword': # weighted keyword aggregation
        model = WeightedKeywordAgg(llm, relative_threshold=args.alpha, absolute_threshold=args.beta, gamma=args.gamma, longgen=longgen) 
    elif args.defense_method == 'decoding':
        if args.eta>0 and not longgen:
            logger.warning(f"using non-zero eta {args.eta} for QA")
        model = WeightedDecodingAgg(llm, eta=args.eta, gamma=args.gamma)
    elif args.defense_method == 'graph':
        model = GraphBasedRRAG(llm)
    elif args.defense_method == 'MIS':
        model = MISBasedRRAG(llm)
    elif args.defense_method == "sampling":
        model = RandomSamplingReQueryAgg(
            llm=llm,
            sample_size=args.m,
            num_samples=args.T,
            gamma=args.gamma,
        )
    elif args.defense_method == "sampling_keyword":
        model = SamplingWithKeyWordAggregation(
            llm=llm,
            sample_size=args.m,
            num_samples=args.T,
            gamma=args.gamma,
            relative_threshold=args.alpha,
            absolute_threshold=args.beta,
            abstention_threshold=1,
        )
    elif args.defense_method == 'instructrag_icl':
        model = InstructRAG_ICL(llm)
    elif args.defense_method == 'astuterag':
        model = AstuteRAG(llm)

    no_attack = args.attack_method == 'none' or args.top_k<=0 # do not run attack

    # INSTANTIATE ATTACKER
    if no_attack:
        pass
    elif args.attack_method == 'PIA':
        if args.dataset_name == 'biogen':
            attacker = PIALONG(top_k = args.top_k, repeat=3, poison_pos = args.attackpos)
        else:
            attacker = PIA(top_k = args.top_k, repeat=10, poison_pos = args.attackpos)
    elif args.attack_method == 'Poison':
        if args.dataset_name == 'biogen':
            attacker = PoisonLONG(top_k = args.top_k, repeat=3, poison_pos = args.attackpos)
        else:
            attacker = Poison(top_k = args.top_k, repeat=10, poison_pos = args.attackpos)
    else:
        raise NotImplementedError(f"Attack method {args.attack_method} is not implemented.")
    
    # can limit number of samples (for debugging)
    data_list = data_tool.data
    if args.max_samples is not None:
        data_list = data_list[:args.max_samples]

    output_csv_file = f"./output/{LOG_NAME}.csv"
    fieldnames = [
        "rep_idx",
        "acc",
        "asr",
        "initial_acc",
        "initial_asr",
        "input_tokens",
        "output_tokens",
        "total_time_sec",
        "defense_method",
        "gamma",
        "attack_method",
        "attackpos",
        "model_name",
        "dataset_name",
        "dataset_size",        
    ]
    with open(output_csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    llm_judge = LLMJudge() if args.defense_method in ['astuterag', 'instructrag_icl'] else None
    
    response_list = []
    for rep_idx in range(args.rep):
        corr_cnt = 0
        asr_cnt = 0
        input_tokens = 0
        output_tokens = 0
        total_time = 0

        # asr and corr before post-processing with llm-judge (for astuterag and instructrag)
        initial_corr_cnt = 0 
        initial_asr_cnt = 0

        for data_idx, data_item in enumerate(tqdm(data_list)):
            logger.info(f'==== rep_idx #{rep_idx}; item: {data_idx} ====')
    
            # ADD ATTACK TO DATA ITEM 
            data_item = data_tool.process_data_item(data_item)
            query = data_item["question"]
            if not no_attack:
                data_item = attacker.attack(data_item)

            # APPLY DEFENSE
            start_time = time.perf_counter()
            llm.reset_token_count()
            if args.defense_method == "none":
                response = model.query_undefended(data_item)
            elif args.defense_method in ["graph", "MIS", "astuterag", "instructrag_icl", "voting", "keyword", "decoding", "sampling", "sampling_keyword"]:
                response = model.query(data_item)
            else:
                raise NotImplementedError(f"Defense method {args.defense_method} is not implemented.")
            end_time = time.perf_counter()
            token_count = llm.get_token_count()
            
            # EVALUATE RESPONSE
            if args.defense_method in ['astuterag', 'instructrag_icl']:
                # for astuterag and instructrag we post-process the response with llm-judge
                final_response = llm_judge.judge(query, response)
            else:
                final_response = response

            corr = data_tool.eval_response(final_response, data_item)
            corr_cnt += corr
            if not no_attack:
                asr = data_tool.eval_response_asr(final_response, data_item)
                asr_cnt += asr

            # get asr and corr before llm-judge post-processing (for astuterag and instructrag)
            initial_corr_cnt += data_tool.eval_response(response, data_item)
            if not no_attack:
                initial_asr_cnt += data_tool.eval_response_asr(response, data_item)
    
            response_list.append({
                "query": query,
                "initial_response": response,
                "final_response": final_response,
                "defense": args.defense_method,
                "rep": rep_idx,
                "answer":  data_item['answer'], 
                "incorrect_answer": data_item['incorrect_answer']
            })
            input_tokens += token_count["input"]
            output_tokens += token_count["output"]
            llm.reset_token_count()
            total_time += (end_time - start_time)

        logger.info(f'Result for rep: {rep_idx}')
        logger.info(f'corr_cnt: {corr_cnt} out of {len(data_list)}')
        logger.info(f'asr_cnt: {asr_cnt} out of {len(data_list)}')

        # save for later analysis, currently used for biogen dataset 
        if args.save_response:
            os.makedirs(f'result/',exist_ok=True)
            with open(f'result/{LOG_NAME}.json','w') as f:
                json.dump(response_list,f,indent=4)

        if args.use_cache:
            llm.dump_cache()

        result_current = {
            "rep_idx": rep_idx,
            "acc": corr_cnt / (len(data_list)),
            "asr": asr_cnt / len(data_list),
            "initial_acc": initial_corr_cnt / (len(data_list)),
            "initial_asr": initial_asr_cnt / len(data_list),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_time_sec": round(total_time, 2),
            "defense_method": args.defense_method,
            "gamma": args.gamma,
            "attack_method": args.attack_method,
            "attackpos": args.attackpos,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "dataset_size": len(data_list),
        }
        df = pd.DataFrame([result_current])
        df.to_csv(output_csv_file, mode='a', header=False, index=False)

if __name__ == '__main__':
    main()