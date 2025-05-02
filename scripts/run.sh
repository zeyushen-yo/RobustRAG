#!/bin/bash
declare -a defenses=( "keyword" "none" )
declare -a datasets=( "realtimeqa_sorted" "open_nq_sorted" "triviaqa_sorted" )
declare -a models=( "gpt-4o" )
declare -a attacks=( "PIA" "Poison" "none" )
declare -a attackpositions=( 0 9 )
declare -a gammas=( 1 )
for defense in "${defenses[@]}"; do
    for data in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            for attack in "${attacks[@]}"; do
                for attackpos in "${attackpositions[@]}"; do
                    for gamma in "${gammas[@]}"; do
                        echo "executing $model-$data-$defense-$attack-$attackpos-$gamma"
                        sbatch scripts/run_cpu.slurm $defense $data $model $attack $attackpos $gamma
                    done
                done
            done
        done
    done
done