# Script to reproduce bert-base with the same setup
# Train for ~25k steps with 4 Titan-RTX gpus

export WANDB_MODE=online
deepspeed --num_gpus 4 run_pretraining.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 768 \
  --num_hidden_layers 12 \
  --num_attention_heads 12 \
  --intermediate_size 3072 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr 1e-3 \
  --train_batch_size 4096 \
  --train_micro_batch_size_per_gpu 256 \
  --lr_schedule step \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps 25000 \
  --num_warmup_steps 1500 \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10000 \
  --dataset_path /n/tata_ddos_ceph/woojeong/data/enwiki_books_128_20/total_balanced/ \
  --output_dir /n/tata_ddos_ceph/woojeong/saved_models/pretrain/ \
  --job_name base \
  --current_run_id default-5val \
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
  --seed 42 \
  --fp16 \
  --load_tokenizer_locally