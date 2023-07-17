# profile vanilla bert models, change hidden size, num_attention_heads, intermediate_size
# dataset path should be `/home/wk247/data/enwiki_books_128_20_ver3/profile/`

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export KINETO_LOG_LEVEL=3
python profile_model.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 512 \
  --num_hidden_layers 24 \
  --num_attention_heads 8 \
  --intermediate_size 2048 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode post-ln \
  --lr 2e-4 \
  --train_batch_size 32 \
  --train_micro_batch_size_per_gpu 32 \
  --lr_schedule step \
  --curve linear \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps 50000 \
  --num_warmup_steps 1250 \
  --warmup_proportion 0.025 \
  --print_steps 100 \
  --dataset_path /home/wk247/data/enwiki_books_128_20_ver3/profile/ \
  --output_dir /home/wk247/saved_models/pretrain-2steps/ \
  --job_name halflarge-hf \
  --current_run_id set2-bsz512-50ksteps-1.2kwarmup-5val-lr2e-4 \
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