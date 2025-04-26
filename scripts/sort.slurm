#!/bin/bash
#SBATCH --job-name=RobustRAGSort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G  
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00 
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=zs7353@princeton.edu
module purge
module load proxy/default anaconda3/2024.2 gcc-toolset/13 cudatoolkit/12.6
conda activate RobustRAG

export AI_SANDBOX_KEY="ae696dffa9874e17bbd7f13499d3b571"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/zs7353/RobustRAG/data/sort.py