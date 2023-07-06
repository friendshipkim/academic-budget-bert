#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate pretrain
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_P2P_LEVEL=NVL
deepspeed --num_gpus 1 --master_port 29503 run_pretraining.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 512 \
  --num_hidden_layers 24 \
  --num_attention_heads 8 \
  --intermediate_size 2048 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode post-ln \
  --lr 8e-5 \
  --train_batch_size 512 \
  --train_micro_batch_size_per_gpu 64 \
  --lr_schedule step \
  --curve linear \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps 160000 \
  --num_warmup_steps 0 \
  --warmup_proportion 0.0 \
  --print_steps 100 \
  --num_epochs_between_checkpoints 100 \
  --dataset_path /home/wk247/data/enwiki_books_128_20_ver2/set23/ \
  --output_dir /home/wk247/saved_models/ligo-bert-pretrain/ \
  --job_name 2xhalflarge-hf \
  --current_run_id set23-160ksteps-nowarmup-bsz512-lr8e-5-noavg-5val \
  --project_name budget-bert-pretraining \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 64 \
  --validation_shards 5 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --seed 333 \
  --fp16 \
  --hf_architecture \
  --num_src_models 2 \
  --avg_logits True \
  --finetuned_model_path /home/wk247/saved_models/ligo-bert/2xhalflarge-hf-set23-100steps-nowarmup-bsz512-lr1e-5-eyeinit-noavg-val20-base512-2e-4/set23-100steps-nowarmup-bsz512-lr1e-5-eyeinit-noavg-val20-base512-2e-4/epoch1_step100/