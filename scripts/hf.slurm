#!/bin/bash
#SBATCH --job-name=RobustRAG-llama-rqa-keyword-poison
#SBATCH --nodes=1  
#SBATCH --ntasks=1
#SBATCH --mem=30G  
#SBATCH --gres=gpu:1   
#SBATCH --time=12:00:00
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=bi0600@princeton.edu
module purge
module load anaconda3/2024.2
conda activate RobustRAG

# attack
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison 

# benign
#python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method none 
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method none 

#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method none --debug
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA --debug
#python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison --debug

#python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method none --debug
#python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA --debug
#python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison --debug
python temp.py --model_name mistral7b --dataset_name realtimeqa --rep 1 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison --debug
