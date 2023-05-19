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

from run_pretraining import parse_arguments

# logger = Logger(cuda=torch.cuda.is_available())
# logging.basicConfig(filename="./finetune_100step_nowarmup_lr2e-4.log", filemode='w', level=logging.INFO)
device = "cuda:0" if torch.cuda.is_available() else "cpu"


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
    # src1_model: 1.9404
    # src2_model: 1.9360
    src_model_list, target_model = init_ligo(args)
    model = target_model
    model.network.to(device)
    # breakpoint()
    
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
    wandb.init(project=args.project_name, group="ligo_2xhalflarge", name=args.current_run_id, dir="/tmp")
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
            

if __name__ == "__main__":
    main()
