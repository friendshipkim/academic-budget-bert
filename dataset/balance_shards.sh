#!/bin/bash

for DIR_NAME in $(ls -d /n/tata_ddos_ceph/woojeong/data/output/train_128_20_*)
do
    if [[ -d ${DIR_NAME}_balanced ]] || [[ $DIR_NAME =~ "balanced" ]]
    then
        echo "Directory $DIR_NAME exists, skipping" 
    else
        echo "Balance train shards in $DIR_NAME"
        python balance_shards.py \
            --dir ${DIR_NAME} \
            --out-dir ${DIR_NAME}_balanced
    fi
done

for DIR_NAME in $(ls -d /n/tata_ddos_ceph/woojeong/data/output/test_*)
do
    if [[ -d ${DIR_NAME}_balanced ]] || [[ $DIR_NAME =~ "balanced" ]]
    then
        echo "Directory $DIR_NAME exists, skipping" 
    else
        echo "Balance test shards in $DIR_NAME"
        python balance_shards.py \
            --dir ${DIR_NAME} \
            --out-dir ${DIR_NAME}_balanced
    fi
done

