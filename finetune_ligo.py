import torch
import wandb
import logging
import glob
import os
from tqdm import tqdm

from pretraining.base import BasePretrainModel
from pretraining.schedules import get_scheduler
from pretraining.ligo_register_utils import register_models, check_tied_weights
from pretraining.ligo_remove_utils import remove_models
from pretraining.dataset.pretraining_dataset import (
    PreTrainingDataset,
    ValidationDataset,
)

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torch.nn import CrossEntropyLoss

from run_pretraining import parse_arguments

# logger = Logger(cuda=torch.cuda.is_available())
# logging.basicConfig(filename="./finetune_100step_nowarmup_lr2e-4.log", filemode='w', level=logging.INFO)
device = "cuda" if torch.cuda.is_available() else "cpu"


def count_parameterized_parameters(model):
    return sum(p.numel() for n, p in model.named_parameters() if (p.requires_grad) and ("original" not in n))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_ligo(args):
    """
    Prepare ligo from the source model
    - source model paths: args.src_model*_path
    """
    # save original avg_logits
    avg_logits = args.avg_logits
    
    # Load two pre-training model skeletons + supplied model config
    src_model_list = []
    for src_index in range(1, args.num_src_models + 1):
        src_model_path = eval(f"args.src_model{src_index}_path")
        logging.info(f"Loading source model {src_index} from {src_model_path}")
        
        # base model logits are not averaged
        args.avg_logits = False
        src_model = BasePretrainModel(args)
        
        # checkpoint: OrderedDict with model params
        checkpoint = torch.load(os.path.join(src_model_path, "pytorch_model.bin"))
        src_model.network.load_state_dict(checkpoint)
        src_model_list.append(src_model)
    
    # stitched model skeleton
    args.avg_logits = avg_logits
    logging.info(f"Initializing ligo model with {args.num_src_models} models...")
    logging.info(f"Tie weights: {not args.untie_weights}, Avg logits: {args.avg_logits}, Init_type: {args.init_type}")
    ligo_stitched_model = BasePretrainModel(args, model_type="ligo-stitched-bert-mlm")
    
    register_models(
        tgt_model=ligo_stitched_model.network,
        src_model_list=[src_model.network for src_model in src_model_list],
        untie_weights=args.untie_weights,
        init_type=args.init_type,
    )
    
    # # check if weights are tied properly
    # check_tied_weights(ligo_stitched_model.network)
    
    # # delete source models
    # del src_model_list

    return src_model_list, ligo_stitched_model


def get_valid_dataloader(args, dataset: Dataset):
    return (x for x in DataLoader(
        dataset,
        batch_size=args.validation_micro_batch,
        sampler=RandomSampler(dataset),
        num_workers=0,
        pin_memory=True,
    ))


# global index to fetch a validation shard from the dataset
validation_shard_index = 0


def pretrain_validation(args, model, validation_dataset, shard_index=0):
    global validation_shard_index

    logging.info(f"Validation micro batch size: {args.validation_micro_batch}")
    index = validation_shard_index
    validation_shard_index += 1
    model.eval()
    dataset = validation_dataset.get_validation_set(index)
    data_batches = get_valid_dataloader(args, dataset)
    eval_loss = 0
    num_eval_steps = 0
    for _, batch in enumerate(tqdm(data_batches, smoothing=1)):
        batch = tuple(t.to(device) for t in batch)
        total_loss = model.forward(batch)
        eval_loss += total_loss.item()
        num_eval_steps += 1
    eval_loss = eval_loss / num_eval_steps

    logging.info(f"Validation Loss for shard {shard_index} is: {eval_loss}")

    del dataset
    del data_batches
    del batch
    model.train()
    return eval_loss


def report_train_metrics(current_lr, loss, step, data_sample_count):
    log_info = {
        "train/lr": current_lr,
        "train/train_loss": loss,
    }
    wandb.log(log_info, step=step)
    samp_info = {
        "Train/Samples/train_loss": loss,
        "Train/Samples/lr": current_lr,
        "Train/total_samples": data_sample_count,
    }
    wandb.log(samp_info, commit=False)
    

def main():
    args = parse_arguments()
    
    # register ligo parameterization
    _, model = init_ligo(args)
    model.network.to(device)
    
    # optimizer, lr scheduler
    # NOTE: ligo params of ln and bias are tied, so no need to group them
    # optimizer_grouped_parameters = model.prepare_optimizer_parameters(args.optimizer_args.weight_decay)
    optimizer = AdamW(
        model.network.parameters(),
        lr=args.lr,
        betas=(args.optimizer_args.adam_beta1, args.optimizer_args.adam_beta2),
        eps=args.optimizer_args.adam_eps,
        weight_decay=args.optimizer_args.weight_decay,
    )
    optimizer.zero_grad()
    lr_scheduler = get_scheduler(args.schedule_args, optimizer, args)
    
    # setup W&B logging
    wandb.init(project=args.project_name, group=args.group_name, name=args.job_name, dir="/tmp")
    wandb.config.update(args, allow_val_change=True)
    wandb.config.update({
        'weight_decay': args.optimizer_args.weight_decay,
    }, allow_val_change=True)
    wandb.watch(model.network)
    
    # datasets
    pretrain_dataset_provider = PreTrainingDataset(args, logger=args.logger)
    validation_dataset = ValidationDataset(args)
    
    # train
    total_epoch = 10
    data_sample_count = 0
    current_step = 0
    finetune_steps = args.max_steps
    num_accumulation_steps = args.train_batch_size // args.train_micro_batch_size_per_gpu
    
    for epoch in range(total_epoch):
        # dataset
        dataset_iterator, total_length = pretrain_dataset_provider.get_shard(epoch)
        logging.info(f"Epoch: {epoch + 1}")
        model.train()
        scaled_loss = []
        
        for batch_index_number, batch_index in enumerate(tqdm(dataset_iterator, smoothing=1)):
            if batch_index_number >= len(dataset_iterator) - 1:
                # skip last batch
                optimizer.zero_grad()
                continue
            
            # batch
            # (?: [32, 1], input_ids: [32, 128], attention_mask:[32, 128], token_type_ids (all 0): [32, 128], masked_lm_labels: [32, 128])
            batch = pretrain_dataset_provider.get_batch(batch_index)
            batch = tuple(t.to(device) for t in batch)  # Move to GPU
            
            loss = model.forward(batch)
            
            data_sample_count += args.train_micro_batch_size_per_gpu
            
            # Normalize the Gradients
            loss = loss / num_accumulation_steps
            scaled_loss.append(loss.item())
            loss.backward()

            if ((batch_index_number + 1) % num_accumulation_steps == 0):
                # Update Optimizer
                optimizer.step()
                optimizer.zero_grad()
                # Do not update lr scheduler with constant lr
                # lr_scheduler.step()
                
                current_step += 1
                last_lr = lr_scheduler.get_last_lr()[0]
                logging.info(f"step: {current_step}, batch index: {batch_index_number+1}, loss: {sum(scaled_loss)}, lr: {last_lr}")
                report_train_metrics(
                    last_lr,
                    sum(scaled_loss),
                    current_step,
                    data_sample_count
                )
                
                scaled_loss = []
                    
                # save checkpoint
                if len(args.num_steps_between_checkpoints) != 0 and (current_step >= args.num_steps_between_checkpoints[0]):
                    # run validation
                    if args.do_validation:
                        eval_losses = []
                        for shard_index in range(args.validation_shards):
                            eval_loss = pretrain_validation(args, model, validation_dataset, shard_index)
                            eval_losses.append(eval_loss)
                        eval_loss = sum(eval_losses) / len(eval_losses)
                        logging.info(f"val loss: {eval_loss}")
                        log_info = {
                            "Validation/Loss": eval_loss,
                        }
                        wandb.log(log_info, step=current_step)
                    
                    logging.info(f"Saving checkpoint to {args.saved_model_path}")
                    model.save_weights(
                        checkpoint_id=f"epoch{epoch+1}_step{current_step}",
                        output_dir=args.saved_model_path,
                        is_deepspeed=False,
                    )
                    args.num_steps_between_checkpoints.pop(0)
            
                if current_step == finetune_steps:
                    logging.info("end of finetuning")
                    
                    # run validation
                    if args.do_validation:
                        eval_losses = []
                        for shard_index in range(args.validation_shards):
                            eval_loss = pretrain_validation(args, model, validation_dataset, shard_index)
                            eval_losses.append(eval_loss)
                        eval_loss = sum(eval_losses) / len(eval_losses)
                        logging.info(f"val loss: {eval_loss}")
                        log_info = {
                            "Validation/Loss": eval_loss,
                        }
                        wandb.log(log_info, step=current_step)
                    
                    # save checkpoint
                    logging.info(f"Saving checkpoint to {args.saved_model_path}")
                    model.save_weights(
                        checkpoint_id=f"epoch{epoch+1}_step{current_step}",
                        output_dir=args.saved_model_path,
                        is_deepspeed=False,
                    )
                    
                    # save non-parameterized models under args.saved_model_path
                    del dataset_iterator
                    del model
                    logging.info("Saving non-parameterized models to 'args.saved_model_path/removed'")
                    save_removed_models(args, sanity_batch=batch)
                    return


def save_removed_models(args=None, sanity_batch=None):
    """
    Save removed models after sanity check on the batch
    Sanity check: Pamaterized model -> removed model: loss should match
    """
    if args is None:
        # when executing this method separately
        args = parse_arguments()
        pretrain_dataset_provider = PreTrainingDataset(args, logger=args.logger)
        dataset_iterator, total_length = pretrain_dataset_provider.get_shard(0)
        for batch_index_number, batch_index in enumerate(tqdm(dataset_iterator, smoothing=1)):
            batch = pretrain_dataset_provider.get_batch(batch_index)
            batch = tuple(t.to(device) for t in batch)  # Move to GPU
            sanity_batch = batch
            break
    
    saved_param_models = glob.glob(f"{args.saved_model_path}/*")
    logging.info(f"Found {len(saved_param_models)} saved parameterized models")
    for i, param_model_path in enumerate(saved_param_models):
        logging.info(f"Processing {i+1}-th parameterized model...")
        # define and load parameterized model
        src_model_list = [BasePretrainModel(args) for _ in range(args.num_src_models)]
        param_model = BasePretrainModel(args, model_type="ligo-stitched-bert-mlm")
        register_models(
            tgt_model=param_model.network,
            src_model_list=[src_model.network for src_model in src_model_list],
            untie_weights=args.untie_weights,
            init_type=args.init_type,
        )

        logging.info(f"Loading parameterized target model from {param_model_path}")
        param_checkpoint = torch.load(os.path.join(param_model_path, "pytorch_model.bin"))
        param_model.network.load_state_dict(param_checkpoint)
        param_model.network.to(device)
        param_model.eval()
        
        with torch.no_grad():
            param_model_loss = param_model.network(sanity_batch)[0].item()
            param_model_size = count_parameters(param_model.network)
        
        # remove parameterization
        logging.info("Removing parameterization")
        remove_models(param_model.network)
        param_model.network.to(device)
        param_model.eval()
        
        with torch.no_grad():
            removed_model_loss = param_model.network(sanity_batch)[0].item()
            removed_model_size = count_parameters(param_model.network)
            
        # sanity check
        logging.info(f"Parameterized model loss: {param_model_loss:.3f}, size: {param_model_size}")
        logging.info(f"Removed model loss: {removed_model_loss:.3f}, size: {removed_model_size}")
        assert param_model_loss == removed_model_loss, "Sanity check failed"
        
        # if sanity check passed, save removed model
        logging.info(f"Sanity check passed, saving non-parameterized model to {param_model_path}/removed")
        param_model.save_weights(
            checkpoint_id="removed",
            output_dir=param_model_path,
            is_deepspeed=False,
        )


def sanity_2models_eyeinit():
    args = parse_arguments()
    args.init_type = 'eye'
    
    # register ligo parameterization
    src_model_list, target_model = init_ligo(args)
    tgt_model = target_model
    src_model_list[0].network.to(device)
    src_model_list[1].network.to(device)
    tgt_model.network.to(device)
    
    tgt_model.eval()
    src_model_list[0].eval()
    src_model_list[1].eval()
    
    # datasets
    pretrain_dataset_provider = PreTrainingDataset(args, logger=args.logger)
    dataset_iterator, total_length = pretrain_dataset_provider.get_shard(0)
    for batch_index_number, batch_index in enumerate(tqdm(dataset_iterator, smoothing=1)):
        batch = pretrain_dataset_provider.get_batch(batch_index)
        batch = tuple(t.to(device) for t in batch)  # Move to GPU
        
        input_ids = batch[1]  # [32, 128]
        attention_mask = batch[2]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [32, 128]
        token_type_ids = batch[3]  # (all 0): [32, 128]
        masked_lm_labels = batch[4]  # [32, 128]
        masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1), as_tuple=False).view(-1)
        
        with torch.no_grad():
            # [32, 128, 512]
            src_emb_out0 = src_model_list[0].network.bert.embeddings(input_ids, token_type_ids, skip_ln_dp=True)
            src_emb_out1 = src_model_list[1].network.bert.embeddings(input_ids, token_type_ids, skip_ln_dp=True)
            
            # [32, 128, 1024]
            tgt_emb_out = tgt_model.network.bert.embeddings(input_ids, token_type_ids, skip_ln_dp=True)
            assert torch.allclose(torch.concat((src_emb_out0, src_emb_out1), dim=-1), tgt_emb_out, atol=1e-6)
        
            # bert layer output
            src_layer_out0 = src_model_list[0].network.bert.encoder.layer[0](src_emb_out0, extended_attention_mask, skip_ln_dp=True)
            src_layer_out1 = src_model_list[1].network.bert.encoder.layer[0](src_emb_out1, extended_attention_mask, skip_ln_dp=True)
            tgt_layer_out = tgt_model.network.bert.encoder.layer[0](tgt_emb_out, extended_attention_mask, skip_ln_dp=True)
            assert torch.allclose(torch.concat((src_layer_out0[0], src_layer_out1[0]), dim=-1), tgt_layer_out[0], atol=1e-6)
            
            # encoder output -> nan at last
            src_enc_out0 = src_model_list[0].network.bert.encoder(src_emb_out0, extended_attention_mask, skip_ln_dp=True)
            src_enc_out1 = src_model_list[1].network.bert.encoder(src_emb_out1, extended_attention_mask, skip_ln_dp=True)
            tgt_enc_out = tgt_model.network.bert.encoder(tgt_emb_out, extended_attention_mask, skip_ln_dp=True)
            
            # cls predictions transform
            # [481, 30528]
            src_trans_out0 = src_model_list[0].network.cls.predictions.transform(src_emb_out0, skip_ln_dp=True)
            src_trans_out1 = src_model_list[1].network.cls.predictions.transform(src_emb_out1, skip_ln_dp=True)
            tgt_trans_out = tgt_model.network.cls.predictions.transform(tgt_emb_out, skip_ln_dp=True)
            assert torch.allclose(torch.concat((src_trans_out0, src_trans_out1), dim=-1), tgt_trans_out, atol=1e-6)
            
            # cls predictions
            # [481, 30528]
            src_pred_out0 = src_model_list[0].network.cls(src_emb_out0, masked_token_indexes, skip_ln_dp=True)
            src_pred_out1 = src_model_list[1].network.cls(src_emb_out1, masked_token_indexes, skip_ln_dp=True)
            tgt_pred_out = tgt_model.network.cls(tgt_emb_out, masked_token_indexes, skip_ln_dp=True)
            assert torch.allclose((src_pred_out0 + src_pred_out1) , tgt_pred_out, atol=1e-6)
            
            def _get_loss(prediction_scores):
                loss_fct = CrossEntropyLoss(ignore_index=-1)
                target = torch.index_select(
                    masked_lm_labels.view(-1), 0, masked_token_indexes
                )
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, 30528), target
                )
                return masked_lm_loss
            
            print(f"MLM loss of src model 1: {_get_loss(src_pred_out0).item()}")
            print(f"MLM loss of src model 2: {_get_loss(src_pred_out1).item()}")
            print(f"MLM loss of target model: {_get_loss(tgt_pred_out).item()}")
            
        breakpoint()
        exit()
        
        
def sanity_2models_remove():
    args = parse_arguments()
    
    # # register ligo parameterization
    # _, stitched_model = init_ligo(args)
    # stitched_model.network.to(device)
    
    src_model_list = [BasePretrainModel(args) for _ in range(args.num_src_models)]
    
    logging.info(f"Initializing parameterized model with {args.num_src_models} models...")
    param_model = BasePretrainModel(args, model_type="ligo-stitched-bert-mlm")
    register_models(
        tgt_model=param_model.network,
        src_model_list=[src_model.network for src_model in src_model_list],
        untie_weights=args.untie_weights,
        init_type=args.init_type,
    )

    param_model_path = "/home/wk247/saved_models/ligo-bert/2xhalflarge-hf-set23-100steps-nowarmup-bsz512-lr1e-5-eyeinit-notie-val20-base512-2e-4/set23-100steps-nowarmup-bsz512-lr1e-5-eyeinit-notie-val20-base512-2e-4/epoch1_step100"
    logging.info(f"Loading parameterized target model from {param_model_path}")
    param_checkpoint = torch.load(os.path.join(param_model_path, "pytorch_model.bin"))
    param_model.network.load_state_dict(param_checkpoint)
    param_model.eval()
    param_model.network.to(device)
    
    if not os.path.exists(os.path.join(param_model_path, "removed")):
        logging.info("Removing parameterization")
        remove_models(param_model.network)
        logging.info(f"Saving non-parameterized model to {param_model_path}/removed")
        param_model.save_weights(
            checkpoint_id="removed",
            output_dir=param_model_path,
            is_deepspeed=False,
        )
        exit()
    
    logging.info("Loading removed target model")
    removed_model = BasePretrainModel(args, model_type="stitched-bert-mlm")
    removed_checkpoint = torch.load(os.path.join(param_model_path, "removed/pytorch_model.bin"))
    removed_model.network.load_state_dict(removed_checkpoint)
    removed_model.eval()
    removed_model.network.to(device)
    
    # datasets
    pretrain_dataset_provider = PreTrainingDataset(args, logger=args.logger)
    dataset_iterator, total_length = pretrain_dataset_provider.get_shard(0)
    for batch_index_number, batch_index in enumerate(tqdm(dataset_iterator, smoothing=1)):
        batch = pretrain_dataset_provider.get_batch(batch_index)
        batch = tuple(t.to(device) for t in batch)  # Move to GPU
        
        input_ids = batch[1]  # [32, 128]
        attention_mask = batch[2]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [32, 128]
        token_type_ids = batch[3]  # (all 0): [32, 128]
        masked_lm_labels = batch[4]  # [32, 128]
        masked_token_indexes = torch.nonzero((masked_lm_labels + 1).view(-1), as_tuple=False).view(-1)
        
        # check if param_model's decoder and word embeddings are tied
        word_weight_0 = param_model.network.bert.embeddings.word_embeddings.parametrizations.weight[0].src_weight_0.detach()  # [30528, 512]
        decoder_weight_0 = param_model.network.cls.predictions.decoder.parametrizations.weight[0].src_weight_0.detach()  # [30528, 512]
        
        word_weight_1 = param_model.network.bert.embeddings.word_embeddings.parametrizations.weight[0].src_weight_1.detach()
        decoder_weight_1 = param_model.network.cls.predictions.decoder.parametrizations.weight[0].src_weight_1.detach()
        
        assert torch.allclose(word_weight_0, decoder_weight_0)
        assert torch.allclose(word_weight_1, decoder_weight_1)
        
        # check if ligo params of word embeddings and decoder is tied
        word_ligo_b_0 = param_model.network.bert.embeddings.word_embeddings.parametrizations.weight[0].ligo_b[0].detach()  # [1024, 512]
        decoder_ligo_a_0 = param_model.network.cls.predictions.decoder.parametrizations.weight[0].ligo_a[0].detach()  # [1024, 512]
        
        word_ligo_b_1 = param_model.network.bert.embeddings.word_embeddings.parametrizations.weight[0].ligo_b[1].detach()  # [1024, 512]
        decoder_ligo_a_1 = param_model.network.cls.predictions.decoder.parametrizations.weight[0].ligo_a[1].detach()  # [1024, 512]
        
        assert torch.allclose(word_ligo_b_0, decoder_ligo_a_0)
        assert torch.allclose(word_ligo_b_1, decoder_ligo_a_1)
        
        # check if output weights are the same
        word_weight = torch.mm(word_weight_0, word_ligo_b_0.T) + torch.mm(word_weight_1, word_ligo_b_1.T)
        decoder_weight = torch.mm(decoder_weight_0, decoder_ligo_a_0.T) + torch.mm(decoder_weight_1, decoder_ligo_a_1.T)
        
        assert torch.allclose(word_weight, decoder_weight)
        
        # check if removed weights are the same
        word_weight_removed = removed_model.network.bert.embeddings.word_embeddings.weight.detach()
        decoder_weight_removed = removed_model.network.cls.predictions.decoder.weight.detach()
        
        assert torch.allclose(word_weight, word_weight_removed)
        assert torch.allclose(decoder_weight, decoder_weight_removed)
        
        # check outputs (masked_lm_loss, sequence_output, prediction_scores)
        param_output = param_model.network(batch)
        removed_output = removed_model.network(batch)
        
        for i in range(3):
            assert torch.allclose(param_output[i], removed_output[i])
            
        breakpoint()
        
        # result
        # 2xhalflarge-hf-set23-100steps-nowarmup-bsz512-lr1e-5-eyeinit-noavg-val20-base512-2e-4 - pass
        # 2xhalflarge-hf-set23-100steps-nowarmup-bsz512-lr1e-5-eyeinit-notie-val20-base512-2e-4 - weights are not averaged

        exit()


if __name__ == "__main__":
    main()
    # sanity_2models_remove()
