#! /usr/bin/env bash

lrs=(0.0015 0.001 0.0008)
max_steps=(10000)
warmup_fraction=(0.2 0.225 0.3)
wds=(0.01 0.05 0.001)
dataset_paths=(/opt/ml/data/set0/ /opt/ml/data/set1/ /opt/ml/data/set2/ /opt/ml/data/set3/)

for seed in 42; do 
    for lr in "${lrs[@]}"; do
        for max_step in "${max_steps[@]}"; do
            for warmup in "${warmup_fraction[@]}"; do
                for wd in "${wds[@]}"; do
                    for dataset_path in "${dataset_paths[@]}"; do
                        warmupsteps=$(echo "($warmup * $max_step) / 1" | bc)
                        echo warmupsteps: $warmupsteps
                        dp=`basename $dataset_path`
                        argo submit pretrain.yaml \
``                            --name "pretrain-${seed}-${lr}-${max_step}-${warmup}-${wd}-${dp}" \
                            -p seed=$seed \
                            -p lr=$lr \
                            -p steps=$max_step \
                            -p warmup=$warmupsteps \
                            -p wd=$wd \
                            -p dataset_path=$dataset_path
                            
                    done
                done
                sleep 5 # to avoid argo throttling
            done
        done
    done
done