#! /bin/bash
# Script to pretrain a half-large model
# Train for 10k steps with 1 gpu
OTHER_PARAMS=${@:1}
export WANDB_API_KEY=641959d1c0dbfc348e2e0b75279abe93425c6ec7

export WANDB_MODE=online
deepspeed --include localhost:0 --master_port 29500 run_pretraining.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 512 \
  --num_hidden_layers 24 \
  --num_attention_heads 8 \
  --intermediate_size 2048 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr 0.0015 \
  --train_batch_size 4096 \
  --train_micro_batch_size_per_gpu 1024 \
  --lr_schedule constant_step \
  --curve linear \
  --warmup_proportion 0.25 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.05 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps 10000 \
  --num_warmup_steps 600 \
  --dataset_path /opt/ml/data/set0/ \
  --output_dir /opt/ml/data/saved_models/pretrain/ \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10 \
  --job_name halflarge \
  --current_run_id "$(($RANDOM%256))" \
  --project_name budget-bert-pretraining \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 256 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --seed 42 \
  --fp16 \
  --validation_shards 5 \
   $OTHER_PARAMS