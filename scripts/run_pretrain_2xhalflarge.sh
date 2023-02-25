# Script to train a 2xhalflarge models
# Stitch two half-large models trained on set 0/1
# and train the stitched model on set 2/3 
# Train for ~6k steps with 1 Titan-RTX gpu

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
  --train_micro_batch_size_per_gpu 128 \
  --lr_schedule constant_step \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps 10000 \
  --num_warmup_steps 600 \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10000 \
  --dataset_path /n/tata_ddos_ceph/woojeong/data/enwiki_books_128_20/set23/ \
  --output_dir /n/tata_ddos_ceph/woojeong/saved_models/pretrain/ \
  --job_name 2xhalflarge \
  --current_run_id set23-10ksteps \
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
  --seed 42 \
  --fp16 \
  --load_tokenizer_locally \
  --do_stitch \
  --src_model1_path /n/tata_ddos_ceph/woojeong/saved_models/pretrain/halflarge-0/0/epoch1000000_step10102/ \
  --src_model2_path /n/tata_ddos_ceph/woojeong/saved_models/pretrain/halflarge-1/1/epoch1000000_step10010/