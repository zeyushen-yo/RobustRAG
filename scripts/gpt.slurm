#!/bin/bash
#SBATCH --job-name=RobustRAG
#SBATCH --nodes=1  
#SBATCH --ntasks=1      
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G   
#SBATCH --time=48:00:00
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=zs7353@princeton.edu
module purge
module load anaconda3/2024.2
conda activate RobustRAG
python main.py --model_name gpt-4o --dataset_name simpleqa --rep 2 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method PIA
python main.py --model_name gpt-4o --dataset_name simpleqa --rep 2 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3 --attack_method Poison
python main.py --model_name gpt-4o --dataset_name simpleqa --rep 2 --top_k 10 --defense_method keyword --alpha 0.3 --beta 3