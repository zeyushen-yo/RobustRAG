#!/bin/bash
#SBATCH --job-name=RobustRAG
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G  
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00  
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=zs7353@princeton.edu
module purge
module load anaconda3/2024.2
conda activate RobustRAG

python main.py --model_name llama3b --dataset_name simpleqa --top_k 10 --rep 1 --defense_method MIS --attack_method PIA  --no_vanilla