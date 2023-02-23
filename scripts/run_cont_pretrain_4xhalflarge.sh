#! /bin/bash
# Script to train a 4xhalflarge models
# Stitch four half-large models trained on set 0/1/2/3
# and train the stitched model on all sets
OTHER_PARAMS=${@:1}
export WANDB_API_KEY=641959d1c0dbfc348e2e0b75279abe93425c6ec7

export WANDB_MODE=online

deepspeed --include localhost:0 --master_port 29500 run_pretraining.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 16 \
  --intermediate_size 4096 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr 0.0015 \
  --train_batch_size 4096 \
  --train_micro_batch_size_per_gpu 256 \
  --lr_schedule constant_step \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.05 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps 5000 \
  --num_warmup_steps 1000 \
  --dataset_path /opt/ml/data/total/ \
  --output_dir /opt/ml/data/saved_models/4stitch-cont \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10 \
  --job_name 4xhalflarge-cont \
  --current_run_id total-cont \
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
  --load_from /opt/ml/data/saved_models/4stitch/4xhalflarge-total/total/epoch1000000_step507/ \
  --validation_shards 5 \
  $OTHER_PARAMS