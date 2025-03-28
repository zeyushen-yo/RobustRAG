#!/bin/bash
#SBATCH --job-name=RobustRAG
#SBATCH --nodes=1  
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=pli
#SBATCH --account=llm_negotiation
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=zs7353@princeton.edu
module purge
module load anaconda3/2024.2
conda activate RobustRAG

# python main.py --model_name deepseek7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA 
# python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --eta 0.0 --attack_method PIA 
# python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --eta 0.0 --attack_method PIA 
# python main.py --model_name mistral7b --dataset_name simpleqa --rep 2 --top_k 10 --defense_method decoding --eta 0.0 --attack_method PIA 
# python main.py --model_name llama3b --dataset_name simpleqa --rep 2 --top_k 10 --defense_method decoding --eta 0.0 --attack_method PIA 
# python main.py --model_name mistral7b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --eta 0.0 --attack_method Poison
# python main.py --model_name llama3b --dataset_name realtimeqa --rep 5 --top_k 10 --defense_method decoding --eta 0.0 --attack_method Poison
# python main.py --model_name mistral7b --dataset_name simpleqa --rep 2 --top_k 10 --defense_method decoding --eta 0.0 --attack_method Poison
python main.py --model_name llama3b --dataset_name simpleqa --rep 2 --top_k 10 --defense_method decoding --eta 0.0 --attack_method Poison