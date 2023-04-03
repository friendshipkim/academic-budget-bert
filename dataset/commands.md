## Obtaining textual data
### Enwiki
From https://dumps.wikimedia.org/enwiki/, download `enwiki-{data}-pages-articles-multistream.xml.bz2` file and unzip with
```
bzip2 -dk filename.bz2
```
### Bookcorpus
Contact me

## Data Processing
```bash
pip install wikiextractor

# enwiki
python process_data.py -f <path_to_xml> -o <output_dir> --type wiki

# bookcorpus
python process_data.py -f <path_to_text_files> -o <output_dir> --type bookcorpus
```

## Sharding
``` bash
python shard_data.py \
    --dir <path_to_text_files> \
    -o <output_dir> \
    --num_train_shards 256 \
    --num_test_shards 64 \
    --frac_test 0.1
```
Move test shards to another directory in order to avoid repeating sampling from test shards

## Samples Generation
### Train (`dup_factor=20`)
```bash
python generate_samples.py \
    --dir /n/tata_ddos_ceph/woojeong/data/sharded/train \
    -o  /n/tata_ddos_ceph/woojeong/data/output/train_128_20_total \
    --dup_factor 20 \
    --seed 42 \
    --vocab_file /n/home05/wk247/workspace/academic-budget-bert/local_cache/bert-large-uncased/vocab.txt \
    --do_lower_case 1 \
    --masked_lm_prob 0.15 \
    --max_seq_length 128 \
    --model_name bert-large-uncased \
    --max_predictions_per_seq 20 \
    --n_processes 20
```

### Test (`dup_factor=1`)
```bash
python generate_samples.py \
    --dir /n/tata_ddos_ceph/woojeong/data/sharded/test_64 \
    -o  /n/tata_ddos_ceph/woojeong/data/output/test \
    --dup_factor 1 \
    --seed 42 \
    --vocab_file /n/home05/wk247/workspace/academic-budget-bert/local_cache/bert-large-uncased/vocab.txt \
    --do_lower_case 1 \
    --masked_lm_prob 0.15 \
    --max_seq_length 128 \
    --model_name bert-large-uncased \
    --max_predictions_per_seq 20 \
    --n_processes 20
```

## Split into disjoint sets (Training)
```bash
python split_data_ver2.py \
    --data_path /n/tata_ddos_ceph/woojeong/data/output \
    --split 4 \
    --n_shards 256
```
### (Optional) Copy files to set01 / set 23
```bash
cp train_128_20_set0/* train_128_20_set1/* train_128_20_set01/
cp train_128_20_set2/* train_128_20_set3/* train_128_20_set23/
```

## Balance data
* Balance training shards in train_128_20_* directories
* Balance testing shards in test directories
* Run this after splitting because this could merge two file sampled from one shard
```bash
bash balance_shards.sh
```

## Copy test sets to the same directory
```bash
bash copy_test_shards.sh
```
Copy balanced training/testing shards to a final destination directory



