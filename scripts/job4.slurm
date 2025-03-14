#!/bin/bash
#SBATCH --job-name=RobustRAG
#SBATCH --nodes=1  
#SBATCH --ntasks=4       
#SBATCH --mem=512G  
#SBATCH --gres=gpu:4   
#SBATCH --time=24:00:00
#SBATCH --partition=pli
#SBATCH --account=llm_negotiation
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=zs7353@princeton.edu
module purge
module load anaconda3/2024.2
conda activate RobustRAG
deepspeed --num_gpu=4 main.py --model_name llama3b --dataset_name realtimeqa --top_k 10 --defense_method keyword --corruption_size 1 --alpha 0.3 --beta 3 --attack_method PIA  