# Running on G2

## Bash prefix
```bash
#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate pretrain
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_P2P_LEVEL=NVL
```

## Master port
set `--master_port` to submit as a sbatch job
```
deepspeed --num_gpus 4 --master_port 29500 run_pretraining.py \
```

## Paths
### Data paths
`/home/wk247/data/enwiki_books_128_20_ver2/set*`

### Model checkpoint paths
* Pretrained models - `/home/wk247/saved_models/pretrain/`
* Ligo finetuned models - `/home/wk247/saved_models/ligo-bert/`

## Etc
* unset `--load_tokenizer_locally`

