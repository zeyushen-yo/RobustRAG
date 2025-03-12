#!/bin/bash
#SBATCH --job-name=RobustRAG
#SBATCH --nodes=1  
#SBATCH --ntasks=1       
#SBATCH --mem=128G  
#SBATCH --gres=gpu:1   
#SBATCH --time=12:00:00
#SBATCH --partition=pli
#SBATCH --account=llm_negotiation
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=zs7353@princeton.edu
module purge
module load anaconda3/2024.2
conda activate RobustRAG
python main.py --model_name deepseek7b --dataset_name realtimeqa --top_k 10 --defense_method decoding --corruption_size 1 --eta 0.0 --attack_method PIA 