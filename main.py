import argparse
import os
import json
from tqdm import tqdm
import torch
import logging
from src.dataset_utils import load_data
from src.models import create_model
from src.defense import *
from src.attack import *
from src.helper import get_log_name
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Robust RAG')

    # LLM settings
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed from distributed launcher")
    parser.add_argument('--model_name', type=str, default='mistral7b',choices=['mistral7b','llama3b','gpt-4o','o1-mini','deepseek7b'],help='model name')
    parser.add_argument('--dataset_name', type=str, default='realtimeqa',choices=['realtimeqa-mc','realtimeqa','open_nq','biogen'],help='dataset name')
    parser.add_argument('--top_k', type=int, default=10,help='top k retrieval')

    # attack
    parser.add_argument('--attack_method', type=str, default='none',choices=['none','Poison','PIA'], help='The attack method to use (Poison or Prompt Injection)')

    # defense
    parser.add_argument('--defense_method', type=str, default='keyword',choices=['none','voting','keyword','decoding','greedy'],help='The defense method to use')
    parser.add_argument('--alpha', type=float, default=0.3, help='keyword filtering threshold alpha')
    parser.add_argument('--beta', type=float, default=3.0, help='keyword filtering threshold beta')
    parser.add_argument('--eta', type=float, default=0.0, help='decoding confidence threshold eta')

    # certifcation
    parser.add_argument('--corruption_size', type=int, default=1, help='The corruption size when considering certification/attack')
    parser.add_argument('--subsample_iter', type=int, default=1, help='number of subsampled responses for decoding certifictaion')
    # long gen certifcation # not really used in the paper
    parser.add_argument('--temperature', type=float, default=1.0, help='The temperature for softmax')

    # other
    parser.add_argument('--debug', action = 'store_true', help='output debugging logging information')
    parser.add_argument('--save_response', action = 'store_true', help='save the results for later analysis')
    parser.add_argument('--use_cache', action = 'store_true', help='save/use cache responses from LLM')
    parser.add_argument('--no_vanilla', action = 'store_true', help='do not run vanilla RAG')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    LOG_NAME = get_log_name(args)
    logging_level = logging.DEBUG if args.debug else logging.INFO
    
    # create folder 
    os.makedirs(f'log',exist_ok=True)
    
    logging.basicConfig(#level=logging_level,
        format=':::::::::::::: %(message)s'
    )

    logger = logging.getLogger('RRAG-main')
    logger.setLevel(level=logging_level)
    logger.addHandler(logging.FileHandler(f"log/{LOG_NAME}.log"))

    logger.info(args)

    device = 'cuda'

    # load data
    data_tool = load_data(args.dataset_name,args.top_k)

    if args.use_cache: # use/save cached responses from LLM
        os.makedirs(f'cache/',exist_ok=True)
        cache_path = f'cache/{args.model_name}-{args.dataset_name}-{args.top_k}.z'
    else:
        cache_path = None

    # create LLM 
    if args.dataset_name == 'biogen':
        llm = create_model(args.model_name,max_output_tokens=500)
        # path for saving certification data
        os.makedirs(f'result_certify',exist_ok=True)
        certify_save_path = f'result_certify/{LOG_NAME}.json'
        longgen = True
    else:
        llm = create_model(args.model_name,cache_path=cache_path)
        certify_save_path = ''
        longgen = False
    no_defense = args.defense_method == 'none' or args.top_k<=0 # do not run defense

    # wrap LLM with the defense class
    if args.defense_method == 'voting': # weighted majority voting
        assert 'mc' in args.dataset_name
        model = WeightedMajorityVoting(llm)
    elif args.defense_method == 'keyword': # weighted keyword aggregation
        model = WeightedKeywordAgg(llm,relative_threshold=args.alpha,absolute_threshold=args.beta,longgen=longgen,certify_save_path=certify_save_path) 
    elif args.defense_method == 'decoding':
        if args.eta>0 and not longgen:
            logger.warning(f"using non-zero eta {args.eta} for QA")
        eval_certify = len(certify_save_path)==0
        model = WeightedDecodingAgg(llm,args,eval_certify=eval_certify,certify_save_path=certify_save_path)
    elif args.defense_method == 'greedy':
        # TODO: change malicious_threshold to a command line argument
        model = GreedyRAG(llm, args, malicious_threshold=0.1)
    else:
        model = RRAG(llm) # base class

    # init attack class
    no_attack = args.attack_method == 'none' or args.top_k<=0 # do not run attack

    if args.defense_method == 'greedy':
        gamma_values = [1] # dummy. gamma is not useful in this case
    else:
        gamma_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    rep = 10

    robustness_all = {gamma: [] for gamma in gamma_values}

    for gamma in gamma_values:
        for i in range(args.top_k):
            if no_attack:
                pass
            elif args.attack_method == 'PIA':
                if args.dataset_name == 'biogen':
                    attacker = PIALONG(top_k = args.top_k, repeat=3, poison_pos = i)
                else:
                    attacker = PIA(top_k = args.top_k, repeat=10, poison_pos = i)
            elif args.attack_method == 'Poison':
                if args.dataset_name == 'biogen':
                    attacker = PoisonLONG(top_k = args.top_k, repeat=3, poison_pos = i)
                else:
                    attacker = Poison(top_k = args.top_k, repeat=10, poison_pos = i)
            else:
                NotImplementedError

            if not no_attack:
                args.corruption_size = 0 # no certification for attack    # ad-hoc implementation -- tofix

            defended_corr_cnt = 0
            undefended_corr_cnt = 0
            certify_cnt = 0
            undefended_asr_cnt = 0
            defended_asr_cnt = 0
            corr_list = []
            response_list = []
            for data_item in tqdm(data_tool.data):
            
                # clean data_item
                data_item = data_tool.process_data_item(data_item)
                # attack
                if not no_attack:
                    data_item = attacker.attack(data_item)
                
                # undefended
                if not args.no_vanilla:
                    for _ in range(rep):
                        response_undefended = model.query_undefended(data_item)
                        undefended_corr = data_tool.eval_response(response_undefended,data_item)
                        undefended_corr_cnt += undefended_corr
                else:
                    response_undefended = ''
                    undefended_corr = False

                # undefended with asr
                if not no_attack:
                    undefended_asr = data_tool.eval_response_asr(response_undefended,data_item)
                    undefended_asr_cnt += undefended_asr
                
                response_list.append({"query":data_item["question"],"undefended":response_undefended})
                
                # defended
                if not no_defense: 
                    for _ in range(rep):
                        if args.defense_method == 'greedy':
                            response_defended,flagg_docs = model.query(data_item)
                            defended_corr = data_tool.eval_response(response_defended,data_item)
                            defended_corr_cnt += defended_corr
                            if not no_attack:
                                defended_asr = data_tool.eval_response_asr(response_defended,data_item)
                                defended_asr_cnt += defended_asr
                            response_list.append({"query":data_item["question"],"defended":response_defended})
                            print(i in flagg_docs)
                        else:                      
                            response_defended,certificate = model.query(data_item, gamma=gamma, corruption_size=args.corruption_size)
                            defended_corr = data_tool.eval_response(response_defended,data_item)
                            defended_corr_cnt += defended_corr
                            certify_cnt += (defended_corr and certificate)
                            if not no_attack:
                                defended_asr = data_tool.eval_response_asr(response_defended,data_item)
                                defended_asr_cnt += defended_asr
                            response_list.append({"query":data_item["question"],"defended":response_defended})
                            corr_list.append(defended_corr and certificate)

            logger.info(f'undefended_corr_cnt: {undefended_corr_cnt}')
            logger.info(f'defended_corr_cnt: {defended_corr_cnt}')
            logger.info(f'certify_cnt: {certify_cnt}')


            if not no_attack:
                logger.info(f'######################## ASR ########################')
                logger.info(f'undefended_asr_cnt: {undefended_asr_cnt}')
                logger.info(f'defended_asr_cnt: {defended_asr_cnt}')


            # save for later analysis, currently used for biogen dataset 
            if args.save_response:
                os.makedirs(f'result/{args.dataset_name}',exist_ok=True)
                if args.defense_method == 'keyword':
                    with open(f'result/{LOG_NAME}.json','w') as f:
                        json.dump(response_list,f,indent=4)
                else:
                    with open(f'result/{LOG_NAME}.json','w') as f:
                        json.dump(response_list,f,indent=4)


            if args.use_cache:
                llm.dump_cache()
            
            robustness_value = defended_asr_cnt / (len(data_tool.data) * rep)
            robustness_all[gamma].append(robustness_value)

    plt.figure(figsize=(10, 6))
    x_values = list(range(1, args.top_k + 1))
    for gamma in gamma_values:
        if args.defense_method == 'greedy':
            plt.plot(x_values, robustness_all[gamma], marker='o')
        else:
            plt.plot(x_values, robustness_all[gamma], marker='o', label=f'γ = {gamma}')
    
    plt.xlabel("Rank", fontsize=14)
    plt.ylabel("Robustness", fontsize=14)
    plt.title("Rank vs Robustness for Different γ Values", fontsize=16)
    plt.xlim(1, args.top_k)
    plt.ylim(0, 1) 
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f"figs/{LOG_NAME}_gamma.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
    