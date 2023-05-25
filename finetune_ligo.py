import torch
import wandb
import logging
from tqdm import tqdm

from pretraining.base import BasePretrainModel
from pretraining.utils import Logger
from pretraining.schedules import get_scheduler
from pretraining.ligo_utils import register_models, check_tied_weights
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_ligo(args):
    """
    Prepare ligo from the source model
    - source model paths: args.src_model*_path
    """
    
    # Load two pre-training model skeletons + supplied model config
    src_model_list = []
    for src_index in range(1, args.num_src_models + 1):
        src_model_path = eval(f"args.src_model{src_index}_path")
        logging.info(f"Loading source model {src_index} from {src_model_path}")
        src_model = BasePretrainModel(args)
        # checkpoint: OrderedDict with model params
        checkpoint = torch.load(src_model_path + "pytorch_model.bin")
        src_model.network.load_state_dict(checkpoint)
        src_model_list.append(src_model)
    
    # stitched model skeleton
    logging.info(f"Initializing ligo model with {args.num_src_models} models...")
    ligo_stitched_model = BasePretrainModel(args, model_type="ligo-stitched-bert-mlm")
    register_models(ligo_stitched_model.network, [src_model.network for src_model in src_model_list])
    # # check if weights are tied properly
    # check_tied_weights(ligo_stitched_model.network)
    
    # delete source models
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
    src_model_list, target_model = init_ligo(args)
    model = src_model_list[0]
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
            print(batch[1])
            
            loss = model.forward(batch)
            print(loss)
            exit()
            
            data_sample_count += args.train_micro_batch_size_per_gpu
            
            # Normalize the Gradients
            loss = loss / num_accumulation_steps
            scaled_loss.append(loss.item())
            loss.backward()
            # model.network.backward(total_loss)

            if ((batch_index_number + 1) % num_accumulation_steps == 0):
                # model.network.step()
                # Update Optimizer
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                
                current_step += 1
                last_lr = lr_scheduler.get_last_lr()[0]
                logging.info(f"step: {current_step}, batch index: {batch_index_number}, loss: {sum(scaled_loss)}, lr: {last_lr}")
                report_train_metrics(
                    last_lr,
                    sum(scaled_loss),
                    current_step,
                    data_sample_count
                )
                
                scaled_loss = []
            
                # run validation
                if args.do_validation and (current_step % 5 == 0):
                    eval_losses = []
                    for shard_index in range(args.validation_shards):
                        eval_loss = pretrain_validation(args, model, validation_dataset, shard_index)
                        eval_losses.append(eval_loss)
                    eval_loss = sum(eval_losses) / len(eval_losses)
                    logging.info(f"val loss: {eval_loss}")
            
                if current_step == finetune_steps:
                    logging.info("end of finetuning")
                    return


def test_ligo_2models():
    args = parse_arguments()
    
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
            assert torch.isclose(torch.concat((src_emb_out0, src_emb_out1), dim=-1), tgt_emb_out).all()
        
            # bert layer output
            src_layer_out0 = src_model_list[0].network.bert.encoder.layer[0](src_emb_out0, extended_attention_mask, skip_ln_dp=True)
            src_layer_out1 = src_model_list[1].network.bert.encoder.layer[0](src_emb_out1, extended_attention_mask, skip_ln_dp=True)
            tgt_layer_out = tgt_model.network.bert.encoder.layer[0](tgt_emb_out, extended_attention_mask, skip_ln_dp=True)
            assert torch.isclose(torch.concat((src_layer_out0[0], src_layer_out1[0]), dim=-1), tgt_layer_out[0]).all()
            
            # encoder output -> nan at last
            src_enc_out0 = src_model_list[0].network.bert.encoder(src_emb_out0, extended_attention_mask, skip_ln_dp=True)
            src_enc_out1 = src_model_list[1].network.bert.encoder(src_emb_out1, extended_attention_mask, skip_ln_dp=True)
            tgt_enc_out = tgt_model.network.bert.encoder(tgt_emb_out, extended_attention_mask, skip_ln_dp=True)
            
            # cls predictions transform
            # [481, 30528]
            src_trans_out0 = src_model_list[0].network.cls.predictions.transform(src_emb_out0)
            src_trans_out1 = src_model_list[1].network.cls.predictions.transform(src_emb_out1)
            tgt_trans_out = tgt_model.network.cls.predictions.transform(tgt_emb_out)
            assert torch.isclose(torch.concat((src_trans_out0, src_trans_out1), dim=-1), tgt_trans_out).all()
            
            # cls predictions
            # [481, 30528]
            src_pred_out0 = src_model_list[0].network.cls(src_emb_out0, masked_token_indexes)
            src_pred_out1 = src_model_list[1].network.cls(src_emb_out1, masked_token_indexes)
            tgt_pred_out = tgt_model.network.cls(tgt_emb_out, masked_token_indexes)
            assert torch.isclose((src_pred_out0 + src_pred_out1) / 2, tgt_pred_out, atol=1e-6).all()
            
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


if __name__ == "__main__":
    # test_ligo_2models()
    main()
