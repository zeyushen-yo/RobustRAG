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
from llm_judge import llm_judge
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

    # defense
    parser.add_argument('--defense_method', type=str, default='keyword',choices=['none','voting','keyword','decoding', 'sampling', 'astuterag', 'instructrag_icl', 'trustrag', 'graph', 'MIS'],help='The defense method to use')
    parser.add_argument('--alpha', type=float, default=0.3, help='keyword filtering threshold alpha')
    parser.add_argument('--beta', type=float, default=3.0, help='keyword filtering threshold beta')
    parser.add_argument('--eta', type=float, default=0.0, help='decoding confidence threshold eta')
    parser.add_argument('--T', type=int, default=3, help='number of samples for sampling method')
    parser.add_argument('--m', type=int, default=5, help='number of docs per sample for sampling method')
    parser.add_argument('--agg', type=str, default="emb", help='method for aggregating responses from multiple samples')

    # long gen certifcation # not really used in the paper
    parser.add_argument('--temperature', type=float, default=1.0, help='The temperature for softmax')

    # other
    parser.add_argument('--debug', action = 'store_true', help='output debugging logging information')
    parser.add_argument('--save_response', action = 'store_true', help='save the results for later analysis')
    parser.add_argument('--use_cache', action = 'store_true', help='save/use cache responses from LLM')
    parser.add_argument('--no_vanilla', action = 'store_true', help='do not run vanilla RAG')
    parser.add_argument('--max_samples', type=int, default=None, help='limit maximum number of samples to run for testing') 
    parser.add_argument('--use_open_model_api', action = 'store_true', help='use local model instead of APIs for open models') 

    args = parser.parse_args()
    return args

def main():
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

    device = 'cuda'

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
    no_defense = args.defense_method == 'none' or args.top_k<=0 # do not run defense

    if args.defense_method in ['astuterag', 'instructrag_icl', 'graph', 'MIS']:
        gamma_values = [1] # dummy. gamma is not useful in this case
    else:
        gamma_values = [0.5, 0.8, 1.0]

    output_csv_file = f"./output/{LOG_NAME}.csv"
    fieldnames = [
        "gamma",
        "rank",
        "undefended_acc",
        "defended_acc",
        "undefended_asr",
        "defended_asr",
        "undefended_input_tokens",
        "undefended_output_tokens",
        "defended_input_tokens",
        "defended_output_tokens",
        "undefended_total_time_sec",
        "defended_total_time_sec"
    ]
    with open(output_csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    results_all = []
    for gamma in gamma_values:
        if args.defense_method == 'voting': # weighted majority voting
            assert 'mc' in args.dataset_name
            model = WeightedMajorityVoting(llm)
        elif args.defense_method == 'keyword': # weighted keyword aggregation
            model = WeightedKeywordAgg(llm, relative_threshold=args.alpha, absolute_threshold=args.beta, longgen=longgen) 
        elif args.defense_method == 'decoding':
            if args.eta>0 and not longgen:
                logger.warning(f"using non-zero eta {args.eta} for QA")
            model = WeightedDecodingAgg(llm, args)
        elif args.defense_method == 'graph':
            model = GraphBasedRRAG(llm)
        elif args.defense_method == 'MIS':
            model = MISBasedRRAG(llm)
        elif args.defense_method == "sampling":
            model = RandomSamplingReQueryAgg(
                llm=llm,
                sample_size=args.m,
                num_samples=args.T,
                gamma=gamma,
            )
        elif args.defense_method == 'instructrag_icl':
            model = InstructRAG_ICL(llm)
        elif args.defense_method == 'astuterag':
            model = AstuteRAG(llm)

        no_attack = args.attack_method == 'none' or args.top_k<=0 # do not run attack

        for i in range(args.top_k):
            if no_attack:
                pass
            elif args.attack_method == 'PIA':
                if args.dataset_name == 'biogen':
                    attacker = PIALONG(top_k = args.top_k, repeat=3, poison_pos = i)
                else:
                    attacker = PIA(top_k = args.top_k, repeat=1, poison_pos = i)
            elif args.attack_method == 'Poison':
                if args.dataset_name == 'biogen':
                    attacker = PoisonLONG(top_k = args.top_k, repeat=3, poison_pos = i)
                else:
                    attacker = Poison(top_k = args.top_k, repeat=1, poison_pos = i)
            else:
                NotImplementedError

            defended_corr_cnt = 0
            undefended_corr_cnt = 0
            undefended_asr_cnt = 0
            defended_asr_cnt = 0

            #response_list = []
            defended_input_tokens = 0
            defended_output_tokens = 0
            undefended_input_tokens = 0
            undefended_output_tokens = 0

            defended_total_time = 0
            undefended_total_time = 0

            data_list = data_tool.data
            if args.max_samples is not None:
                data_list = data_list[:args.max_samples]
            for data_idx, data_item in enumerate(tqdm(data_list)):

                data_item = data_tool.process_data_item(data_item)
                if not no_attack:
                    data_item = attacker.attack(data_item)
                
                # undefended
                if not args.no_vanilla:
                    start_time = time.perf_counter()
                    llm.reset_token_count()
                    for rep_idx in range(args.rep):
                        logger.info(f'==== gamma: {gamma}, attackpos: {i}, item: {data_idx}, vanilla rep: {rep_idx}')
                        response_undefended = model.query_undefended(data_item)
                        undefended_corr = data_tool.eval_response(response_undefended,data_item)
                        undefended_corr_cnt += undefended_corr
                    token_count = llm.get_token_count()
                    undefended_input_tokens += token_count["input"]
                    undefended_output_tokens += token_count["output"]
                    llm.reset_token_count()

                    end_time = time.perf_counter()
                    undefended_total_time += (end_time - start_time)
                else:
                    response_undefended = ''
                    undefended_corr = False

                if not no_attack:
                    undefended_asr = data_tool.eval_response_asr(response_undefended,data_item)
                    undefended_asr_cnt += undefended_asr
                
                #response_list.append({"query":data_item["question"], "undefended":response_undefended})
                
                if not no_defense:
                    start_time = time.perf_counter()
                    llm.reset_token_count()
                    for rep_idx in range(args.rep):
                        logger.info(f'==== gamma: {gamma}, attackpos: {i}, item: {data_idx}, defended rep: {rep_idx}')
                        if args.defense_method == "graph" or args.defense_method == "MIS":
                            response_defended = model.query(data_item)
                        elif args.defense_method == 'astuterag':
                            response_defended = model.query(data_item)
                        elif args.defense_method == 'instructrag_icl':
                            response_defended = llm_judge("gpt-4o", data_item["question"], model.query(data_item))
                        else:              
                            response_defended = model.query(data_item, gamma=gamma)
                        
                        defended_corr = data_tool.eval_response(response_defended,data_item)
                        defended_corr_cnt += defended_corr
                        
                        if not no_attack:
                            defended_asr = data_tool.eval_response_asr(response_defended,data_item)
                            defended_asr_cnt += defended_asr
                        #response_list.append({"query":data_item["question"],"defended":response_defended})
                    token_count = llm.get_token_count()
                    defended_input_tokens += token_count["input"]
                    defended_output_tokens += token_count["output"]
                    llm.reset_token_count()
                    end_time = time.perf_counter()
                    defended_total_time += (end_time - start_time)

            logger.info(f'Params: Gamma: {gamma}, rank: {i}')
            logger.info(f'undefended_corr_cnt: {undefended_corr_cnt}')
            logger.info(f'defended_corr_cnt: {defended_corr_cnt}')

            if not no_attack:
                logger.info(f'######################## ASR ########################')
                logger.info(f'undefended_asr_cnt: {undefended_asr_cnt}')
                logger.info(f'defended_asr_cnt: {defended_asr_cnt}')


            # save for later analysis, currently used for biogen dataset 
            #if args.save_response:
            #    os.makedirs(f'result/{args.dataset_name}',exist_ok=True)
            #    if args.defense_method == 'keyword':
            #        with open(f'result/{LOG_NAME}.json','w') as f:
            #            json.dump(response_list,f,indent=4)
            #    else:
            #        with open(f'result/{LOG_NAME}.json','w') as f:
            #            json.dump(response_list,f,indent=4)


            if args.use_cache:
                llm.dump_cache()

            result_current = {
                "gamma": gamma,
                "rank": i,
                "undefended_acc": undefended_corr_cnt / (len(data_tool.data) * args.rep),
                "defended_acc": defended_corr_cnt / (len(data_tool.data) * args.rep),
                "undefended_asr": undefended_asr_cnt / len(data_tool.data),
                "defended_asr": defended_asr_cnt / (len(data_tool.data) * args.rep),
                "undefended_input_tokens": undefended_input_tokens/args.rep,
                "undefended_output_tokens": undefended_output_tokens/args.rep,
                "defended_input_tokens": defended_input_tokens/args.rep,
                "defended_output_tokens": defended_output_tokens/args.rep,
                "undefended_total_time_sec": round(undefended_total_time/args.rep, 2),
                "defended_total_time_sec": round(defended_total_time/args.rep, 2),
            }
            df = pd.DataFrame([result_current])
            df.to_csv(output_csv_file, mode='a', header=False, index=False)

if __name__ == '__main__':
    main()