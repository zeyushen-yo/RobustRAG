#!/bin/bash
#SBATCH --job-name=RobustRAG
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64G 
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00  
#SBATCH --mail-type=begin       
#SBATCH --mail-type=end         
#SBATCH --mail-user=zs7353@princeton.edu
module purge
module load proxy/default
module load gcc-toolset/10
module load anaconda3/2024.2
conda activate RobustRAG

export AI_SANDBOX_KEY="ae696dffa9874e17bbd7f13499d3b571"
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)
defense=$1
data=$2
model=$3
attack=$4
attackpos=$5
gamma=$6
python main.py --model_name $model --dataset_name $data --top_k 10 --defense_method $defense --gamma $gamma --attack_method $attack --attackpos $attackpos --rep 1 --m 10 --T 1 --debug --model_dir="/scratch/gpfs/zs7353/" --save_response