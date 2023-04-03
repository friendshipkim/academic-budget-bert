#!/bin/bash
BASE_DIR=/n/tata_ddos_ceph/woojeong/data/output
TARGET_BASE_DIR=/n/tata_ddos_ceph/woojeong/data/enwiki_books_128_20_ver2

for SOURCE_DIR in $(ls -d $BASE_DIR/train_128_20_*_balanced)
do
    # make new target directory to copy files
    LAST_DIR=$(basename $SOURCE_DIR)
    if [[ $LAST_DIR =~ train_128_20_(.*)_balanced$ ]]; then
        NEW_DIR=${BASH_REMATCH[1]}
    fi
    TARGET_DIR=$TARGET_BASE_DIR/$NEW_DIR
    # mkdir $TARGET_DIR

    # copy balanced data to the target directory
    echo "Copying training data from $LAST_DIR to $TARGET_DIR"
    # cp $SOURCE_DIR/* $TARGET_DIR

    # # copy test data to the target directory
    echo "Copying test data from $BASE_DIR/test_balanced to $TARGET_DIR"
    # cp $BASE_DIR/test_balanced/* $TARGET_DIR
done

