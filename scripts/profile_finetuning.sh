# profile ligo finetuning
# change target_hidden_size, target_num_attention_heads
# change num_src_models
# untie_weights affects the number of parameters
# dataset path should be `/home/wk247/data/enwiki_books_128_20_ver3/profile/`

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export KINETO_LOG_LEVEL=3
python profile_model.py \
  --model_type bert-mlm --tokenizer_name bert-large-uncased \
  --hidden_act gelu \
  --hidden_size 512 \
  --target_hidden_size 1024 \
  --num_hidden_layers 24 \
  --num_attention_heads 8 \
  --target_num_attention_heads 16 \
  --intermediate_size 2048 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --encoder_ln_mode post-ln \
  --lr 2e-5 \
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
  --max_steps 100 \
  --num_warmup_steps 0 \
  --warmup_proportion 0.00 \
  --dataset_path /home/wk247/data/enwiki_books_128_20_ver3/profile/ \
  --output_dir /home/wk247/saved_models/ligo-bert-2steps/ \
  --job_name sqrtlarge-hf \
  --current_run_id finetune-100steps-nowarmup-bsz512-lr2e-5-eyeinit-notie-avg-avgoverlap-val20-base1e-4 \
  --project_name ligo-finetuning \
  --seed 33 \
  --fp16 \
  --do_stitch \
  --hf_architecture \
  --do_validation False \
  --num_src_models 2 \
  --src_model1_path /home/wk247/saved_models/pretrain-2steps/halflarge-hf-set0-bsz512-200ksteps-5kwarmup-5val-lr1e-4-50ksave/set0-bsz512-200ksteps-5kwarmup-5val-lr1e-4-50ksave/epoch167_step50267 \
  --src_model2_path /home/wk247/saved_models/pretrain-2steps/halflarge-hf-set1-bsz512-200ksteps-5kwarmup-5val-lr1e-4-50ksave/set1-bsz512-200ksteps-5kwarmup-5val-lr1e-4-50ksave/epoch167_step50267 \
  --untie_weights True \
  --avg_logits True \
  --init_type eye