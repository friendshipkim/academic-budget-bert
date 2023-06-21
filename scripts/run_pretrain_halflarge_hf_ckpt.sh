#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate pretrain
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_P2P_LEVEL=NVL
deepspeed --num_gpus 2 --master_port 29501 run_pretraining.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 512 \
  --num_hidden_layers 24 \
  --num_attention_heads 8 \
  --intermediate_size 2048 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode post-ln \
  --lr 1e-4 \
  --train_batch_size 512 \
  --train_micro_batch_size_per_gpu 256 \
  --lr_schedule step \
  --curve linear \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps 200000 \
  --num_warmup_steps 5000 \
  --warmup_proportion 0.025 \
  --print_steps 100 \
  --num_steps_between_checkpoints 50000 \
  --dataset_path /home/wk247/data/enwiki_books_128_20_ver3/set3_500 \
  --output_dir /home/wk247/saved_models/pretrain-2steps/ \
  --job_name halflarge-hf \
  --current_run_id set3-bsz512-200ksteps-5kwarmup-5val-lr1e-4-50ksave \
  --project_name budget-bert-pretraining \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 128 \
  --validation_shards 5 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --seed 33 \
  --fp16 \
  --hf_architecture