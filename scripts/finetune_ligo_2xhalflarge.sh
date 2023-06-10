export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python finetune_ligo.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 512 \
  --num_hidden_layers 24 \
  --num_attention_heads 8 \
  --intermediate_size 2048 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode pre-ln \
  --lr 1e-5 \
  --train_batch_size 512 \
  --train_micro_batch_size_per_gpu 32 \
  --lr_schedule step \
  --curve linear \
  --gradient_clipping 0.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --max_steps 100 \
  --num_warmup_steps 0 \
  --warmup_proportion 0.00 \
  --dataset_path /n/tata_ddos_ceph/woojeong/data/enwiki_books_128_20_ver2/set23/ \
  --output_dir /n/tata_ddos_ceph/woojeong/saved_models/ligo-bert/ \
  --job_name 2xhalflarge \
  --current_run_id set23-100steps-nowarmup-bsz512-lr1e-5-noval-eyeinit \
  --project_name ligo-finetuning \
  --seed 33 \
  --fp16 \
  --load_tokenizer_locally \
  --do_stitch \
  --num_steps_between_checkpoints 10 \
  --num_src_models 2 \
  --src_model1_path /n/tata_ddos_ceph/woojeong/saved_models/pretrain/halflarge-set0-disjoint-bsz4096-10ksteps-5val/set0-disjoint-bsz4096-10ksteps-5val/epoch1000000000_step10024/ \
  --src_model2_path /n/tata_ddos_ceph/woojeong/saved_models/pretrain/halflarge-set1-disjoint-bsz4096-10ksteps-5val/set1-disjoint-bsz4096-10ksteps-5val/epoch1000000000_step10008/