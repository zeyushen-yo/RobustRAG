#!/bin/bash
declare -a defenses=( "MIS" )
declare -a datasets=( "open_nq_sorted" )
declare -a models=( "llama3b" )
declare -a attacks=( "none" )
declare -a attackpositions=( 0 )
declare -a gammas=( 0.9 )
for defense in "${defenses[@]}"; do
    for data in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            for attack in "${attacks[@]}"; do
                for attackpos in "${attackpositions[@]}"; do
                    for gamma in "${gammas[@]}"; do
                        echo "executing $model-$data-$defense-$attack-$attackpos-$gamma"
                        sbatch scripts/run.slurm $defense $data $model $attack $attackpos $gamma
                    done
                done
            done
        done
    done
done