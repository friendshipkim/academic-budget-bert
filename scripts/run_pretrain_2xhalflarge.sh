# Script to train a 2xhalflarge models
# Stitch two half-large models trained on set 0/1 for 10k steps
# and train the stitched model on set 2/3 
# Train for 20k steps with 4 Titan-RTX gpu

export WANDB_MODE=disabled
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# deepspeed --num_gpus 4 run_pretraining.py \
deepspeed --include localhost:0 --master_port 29502 run_pretraining.py \
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
  --train_micro_batch_size_per_gpu 64 \
  --lr_schedule step \
  --curve linear \
  --warmup_proportion 0.06 \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps 5000 \
  --num_warmup_steps 300 \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10000 \
  --dataset_path /n/tata_ddos_ceph/woojeong/data/enwiki_books_128_20/total_balanced/ \
  --output_dir /n/tata_ddos_ceph/woojeong/saved_models/pretrain/ \
  --job_name 2xhalflarge-5ksteps-tmp \
  --current_run_id set23-5ksteps-5val-tmp \
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
  --src_model1_path /n/tata_ddos_ceph/woojeong/saved_models/pretrain/halflarge-set0-10ksteps-5val/set0-10ksteps-5val/epoch1000000_step10022/ \
  --src_model2_path /n/tata_ddos_ceph/woojeong/saved_models/pretrain/halflarge-set1-10ksteps-5val/set1-10ksteps-5val/epoch1000000_step10002/ \
  --record_gradient_norm