export WANDB_MODE=online
# TODO: change lr, dataset_path
# deepspeed --num_gpus 4 run_pretraining.py \
deepspeed --include localhost:0 --master_port 29503 run_pretraining.py \
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
  --train_micro_batch_size_per_gpu 32 \
  --lr_schedule step \
  --curve linear \
  --warmup_proportion 0.0 \
  --gradient_clipping 6.0 \
  --optimizer_type adamw \
  --weight_decay 0.01 \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_eps 1e-6 \
  --total_training_time 48.0 \
  --early_exit_time_marker 48.0 \
  --dataset_path /n/tata_ddos_ceph/woojeong/data/enwiki_books_128_20/set1 \
  --output_dir ./saved_models/training-out-2xhalflarge \
  --print_steps 100 \
  --num_epochs_between_checkpoints 10000 \
  --job_name 2xhalflarge_pretraining \
  --current_run_id 1 \
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
  --use_early_stopping \
  --early_stop_time 180 \
  --early_stop_eval_loss 6 \
  --seed 42 \
  --fp16 \
  --do_stitch \
  --src_model1_path /n/home05/wk247/workspace/academic-budget-bert/saved_models/training-out-halflarge/halflarge_pretraining-0/0 \
  --src_model2_path /n/home05/wk247/workspace/academic-budget-bert/saved_models/training-out-halflarge/halflarge_pretraining-1/1