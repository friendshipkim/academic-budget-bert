#! /bin/bash
# Script to train a 4xhalflarge models
# Stitch four half-large models trained on set 0/1/2/3
# and train the stitched model on all sets
OTHER_PARAMS=${@:1}
export WANDB_API_KEY=641959d1c0dbfc348e2e0b75279abe93425c6ec7

export WANDB_PROJECT=budget-bert-pretraining-test
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
  --lr 1e-3 \
  --train_batch_size 4096 \
  --train_micro_batch_size_per_gpu 256 \
  --lr_schedule constant_step \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps 500 \
  --num_warmup_steps 100 \
  --dataset_path /opt/ml/data/total/ \
  --output_dir /opt/ml/data/saved_models/4stitch \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10 \
  --job_name 4xhalflarge \
  --current_run_id total \
  --project_name budget-bert-pretraining \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 16 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --seed 42 \
  --fp16 \
  --do_stitch \
  --src_model1_path /opt/ml/data/saved_models/halflarge_213-set1-10ksteps/set1-10ksteps/epoch1000000_step10023/ \
  --src_model2_path /opt/ml/data/saved_models/halflarge_146-set1-10ksteps/set1-10ksteps/epoch1000000_step10031/ \
  --src_model3_path /opt/ml/data/saved_models/halflarge_95-set1-10ksteps/set1-10ksteps/epoch1000000_step10005/ \
  --src_model4_path /opt/ml/data/saved_models/halflarge_199-set1-10ksteps/set1-10ksteps/epoch1000000_step10014/ \
  $OTHER_PARAMS
  