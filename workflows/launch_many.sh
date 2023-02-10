#! /usr/bin/env bash

lrs=(0.002 0.001 0.0005 0.0001)
# lrs=(0.002 0.001 0.0005)
max_steps=(5000 10000 15000)
# warmup_fraction=(0 0.06 0.1)
warmup_fraction=(0.02 0.06 0.1 0.2)

for seed in 1; do 
    for lr in "${lrs[@]}"; do
        for max_step in "${max_steps[@]}"; do
            for warmup in "${warmup_fraction[@]}"; do
                warmupsteps=$(echo "($warmup * $max_step) / 1" | bc)
                echo warmupsteps: $warmupsteps
                argo submit finetune.yaml \
                    --name "finetune-${seed}-${lr}-${max_step}-${warmup}" \
                    -p seed=$seed \
                    -p lr=$lr \
                    -p steps=$max_step \
                    -p warmup=$warmupsteps
            done
            sleep 5 # to avoid argo throttling
        done
    done
done