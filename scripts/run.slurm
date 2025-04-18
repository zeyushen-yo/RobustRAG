#!/bin/bash
#SBATCH --job-name=RobustRAG
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G 
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00  
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=bi0600@princeton.edu
module purge
module load proxy/default
module load gcc-toolset/10
module load anaconda3/2024.2
conda activate RobustRAG

python main.py --model_name llama3b --dataset_name realtimeqa --top_k 10 --defense_method astuterag --gamma 1 --attack_method PIA --attackpos 0 --rep 1 --debug --model_dir="/scratch/gpfs/bi0600/" --save_response