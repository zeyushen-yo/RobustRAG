# Table 1
# no RAG and vanilla RAG
#python main.py --model_name mistral7b --dataset_name realtimeqa-mc --rep 5 --top_k 0 --defense_method none  
#python main.py --model_name mistral7b --dataset_name realtimeqa-mc --rep 5 --top_k 10 --defense_method none  
#python main.py --model_name llama3b --dataset_name realtimeqa-mc --rep 5 --top_k 0 --defense_method none  
#python main.py --model_name llama3b --dataset_name realtimeqa-mc --rep 5 --top_k 10 --defense_method none  
#python main.py --model_name gpt-4o --dataset_name realtimeqa-mc --rep 5 --top_k 0 --defense_method none  
#python main.py --model_name gpt-4o --dataset_name realtimeqa-mc --rep 5 --top_k 10 --defense_method none  
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 0 --defense_method none  
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method none  
#python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 0 --defense_method none  
#python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method none  
#python main.py --model_name gpt-4o --dataset_name realtimeqa --rep 5 --top_k 0 --defense_method none  
#python main.py --model_name gpt-4o --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method none  
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 0 --defense_method none  
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method none  
#python main.py --model_name llama3b --dataset_name open_nq --top_k 0 --defense_method none  
#python main.py --model_name llama3b --dataset_name open_nq --top_k 10 --defense_method none  
#python main.py --model_name gpt-4o --dataset_name open_nq --top_k 0 --defense_method none  
#python main.py --model_name gpt-4o --dataset_name open_nq --top_k 10 --defense_method none  

# Voting for RealtimeQA-MC (multiple choice)
#python main.py --model_name mistral7b --dataset_name realtimeqa-mc --rep 5 --top_k 10 --defense_method voting --no_vanilla 
#python main.py --model_name llama3b --dataset_name realtimeqa-mc --rep 5 --top_k 10 --defense_method voting --no_vanilla 
#python main.py --model_name gpt-4o --dataset_name realtimeqa-mc --rep 5 --top_k 10 --defense_method voting --no_vanilla 

# decoding for RealtimeQA and NQ
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --eta 0.0 --no_vanilla 
#python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --eta 0.0 --no_vanilla 
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method decoding --eta 0.0 --no_vanilla 
#python main.py --model_name llama3b --dataset_name open_nq --top_k 10 --defense_method decoding --eta 0.0 --no_vanilla 

# keyword for RealtimeQA and NQ
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --no_vanilla 
#python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --no_vanilla 
#python main.py --model_name gpt-4o --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --no_vanilla 
#python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --no_vanilla 
#python main.py --model_name llama3b --dataset_name open_nq --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --no_vanilla 
#python main.py --model_name gpt-4o --dataset_name open_nq --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --no_vanilla 

# biogen keyword
#python main.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --save_response 
#python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --type pred
#python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --type certify

#python main.py --model_name llama3b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --save_response 
#python llm_eval.py --model_name llama3b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --type pred
#python llm_eval.py --model_name llama3b --dataset_name biogen --top_k 10 --defense_method keyword --alpha 0.4 --beta 4 --type certify

# biogen decoding eta= 4
#python main.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4  --save_response
#python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4  --type pred
#python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4  --type certify

#python main.py --model_name llama3b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4 --save_response
#python llm_eval.py --model_name llama3b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4 --type pred
#python llm_eval.py --model_name llama3b --dataset_name biogen --top_k 10 --defense_method decoding --eta 4 --type certify

# biogen decoding eta=1 with subsampling
#python main.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --save_response
#python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --type pred
#python llm_eval.py --model_name mistral7b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --type certify

#python main.py --model_name llama3b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --save_response
#python llm_eval.py --model_name llama3b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --type pred
#python llm_eval.py --model_name llama3b --dataset_name biogen --top_k 10 --defense_method decoding --eta 1 --type certify


# Figure 3 (top k analysis)
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 2 --defense_method keyword --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 4 --defense_method keyword --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 6 --defense_method keyword --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 8 --defense_method keyword --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 12 --defense_method keyword --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 14 --defense_method keyword --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 16 --defense_method keyword --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 18 --defense_method keyword --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 20 --defense_method keyword --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 2 --defense_method decoding --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 4 --defense_method decoding --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 6 --defense_method decoding --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 8 --defense_method decoding --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 12 --defense_method decoding --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 14 --defense_method decoding --no_vanilla --use_cache 
##python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 16 --defense_method decoding --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 18 --defense_method decoding --no_vanilla --use_cache 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 20 --defense_method decoding --no_vanilla --use_cache 

# Table 2 (empirical attack) 
# python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --eta 0.0 --attack_method PIA 
# python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --eta 0.0 --attack_method Poison 
# python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --eta 0.0 --attack_method PIA 
# python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --eta 0.0 --attack_method Poison 
# python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method decoding --eta 0.0 --attack_method PIA 
# python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method decoding --eta 0.0 --attack_method Poison 
# python main.py --model_name llama3b --dataset_name open_nq --top_k 10 --defense_method decoding --eta 0.0 --attack_method PIA 
# python main.py --model_name llama3b --dataset_name open_nq --top_k 10 --defense_method decoding --eta 0.0 --attack_method Poison 
# python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA 
# python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison 
# python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA 
# python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison 
# python main.py --model_name gpt-4o --dataset_name realtimeqa --rep 2 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA 
# python main.py --model_name gpt-4o --dataset_name realtimeqa --rep 2 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison 
# python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA 
# python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison 
# python main.py --model_name llama3b --dataset_name open_nq --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA 
# python main.py --model_name llama3b --dataset_name open_nq --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison 
# python main.py --model_name gpt-4o --dataset_name open_nq --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA 
# python main.py --model_name gpt-4o --dataset_name open_nq --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison 

# greedy
# python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method greedy --eta 0.0 --attack_method PIA 
# python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method greedy --eta 0.0 --attack_method Poison 
python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method greedy --eta 0.0 --attack_method PIA 
# python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method greedy --eta 0.0 --attack_method Poison 
# python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method greedy --eta 0.0 --attack_method PIA 
# python main.py --model_name mistral7b --dataset_name open_nq --top_k 10 --defense_method greedy --eta 0.0 --attack_method Poison 
# python main.py --model_name llama3b --dataset_name open_nq --top_k 10 --defense_method greedy --eta 0.0 --attack_method PIA 
# python main.py --model_name llama3b --dataset_name open_nq --top_k 10 --defense_method greedy --eta 0.0 --attack_method Poison 