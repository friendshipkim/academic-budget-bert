#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate pretrain
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_P2P_LEVEL=NVL
deepspeed --include localhost:2,3 --master_port 29501 run_pretraining.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 512 \
  --target_hidden_size 768 \
  --num_hidden_layers 24 \
  --num_attention_heads 8 \
  --target_num_attention_heads 12 \
  --intermediate_size 2048 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode post-ln \
  --lr 7e-5 \
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
  --max_steps 50000 \
  --num_warmup_steps 0 \
  --warmup_proportion 0.0 \
  --print_steps 100 \
  --use_early_stopping \
  --early_stop_eval_loss 2.03 \
  --dataset_path /home/azureuser/data/enwiki_books_128_20_ver3/step1_1200/ \
  --output_dir /home/azureuser/saved_models/ligo-bert-pretrain/ \
  --job_name sqrtlarge-resume \
  --current_run_id 1src-50ksteps-nowarmup-bsz512-lr7e-5-5val-base1e-4 \
  --project_name budget-bert-pretraining \
  --validation_epochs 3 \
  --validation_epochs_begin 1 \
  --validation_epochs_end 1 \
  --validation_begin_proportion 0.05 \
  --validation_end_proportion 0.01 \
  --validation_micro_batch 256 \
  --validation_shards 5 \
  --deepspeed \
  --data_loader_type dist \
  --do_validation \
  --seed 33 \
  --fp16 \
  --hf_architecture \
  --num_src_models 1 \
  --avg_decoder True \
  --finetuned_model_path /home/azureuser/saved_models/ligo-bert-2steps/sqrtlarge-hf-1src-finetune-100steps-nowarmup-bsz512-lr2e-5-eyeinit-notie-noavg-val20-base1e-4/finetune-100steps-nowarmup-bsz512-lr2e-5-eyeinit-notie-noavg-val20-base1e-4/epoch1_step100/