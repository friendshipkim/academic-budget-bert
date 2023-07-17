import torch
import logging
import os
from tqdm import tqdm

from torch.optim import AdamW
from pretraining.base import BasePretrainModel
from pretraining.schedules import get_scheduler
from pretraining.utils import count_parameterized_parameters
from pretraining.ligo_register_utils import register_models
from pretraining.dataset.pretraining_dataset import PreTrainingDataset
from run_pretraining import parse_arguments

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    ligo_stitched_model = BasePretrainModel(args, model_type="stitched-bert-mlm")
    
    register_models(
        tgt_model=ligo_stitched_model.network,
        src_model_list=[src_model.network for src_model in src_model_list],
        untie_weights=args.untie_weights,
        init_type=args.init_type,
    )
    
    # # delete source models
    # del src_model_list

    return src_model_list, ligo_stitched_model


def get_info(table):
    syms = [s.strip() for s in table.split("\n")]
    layers = []
    vals = []
    
    # flop_column
    flop_column = [val for val in syms[1].split('  ') if val != ''][-1]
    assert 'FLOPs' in flop_column
    if 'KFLOPS' in flop_column:
        flop_scale = 1e6
    elif 'MFLOPs' in flop_column:
        flop_scale = 1e3
    elif 'GFLOPs' in flop_column:
        flop_scale = 1
    else:
        raise ValueError(f"Unknown flop scale: {flop_column}")

    # sum up flops of all layers
    for row in range(3, 8):
        sym = [val for val in syms[row].split('  ') if val != '']
        if sym[-1] == '--':
            continue
        else:
            layers.append(sym[0].split("::")[-1])
            vals.append(float(sym[-1]))
    
    def _format_to_sec(latency_str):
        if latency_str[-2:] == 'us':
            return float(latency_str[:-2]) / 1e6
        elif latency_str[-2:] == 'ms':
            return float(latency_str[:-2]) / 1e3
        else:
            return float(latency_str[:-1])

    cpu_latency_sec = _format_to_sec(syms[-3][len('Self CPU time total: '):])
    cuda_latency_sec = _format_to_sec(syms[-2][len('Self CUDA time total: '):])
    gflops = sum(vals) / flop_scale
    return cpu_latency_sec, cuda_latency_sec, gflops


def trace_handler(prof):
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    table = prof.key_averages().table(sort_by="flops", row_limit=-1)
    # print(table)
    cpu_latency_sec, cuda_latency_sec, gflops = get_info(table)
    print(f"cpu latency (s): {cpu_latency_sec:.3f}, cuda latency (s): {cuda_latency_sec:.3f}, gflops: {gflops:.3f}")
    exit()


def main():
    args = parse_arguments()
    
    # change args for profiling
    args.train_batch_size = 32
    args.train_micro_batch_size_per_gpu = 32
    
    # if src models are provided, ligo finetuning
    if args.num_src_models >= 0:
        _, model = init_ligo(args)
    # else, base model
    else:
        model = BasePretrainModel(args)
    
    logging.info(f"Model parameters: {count_parameterized_parameters(model.network)}")
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
    
    # datasets
    pretrain_dataset_provider = PreTrainingDataset(args, logger=args.logger)
    
    # train
    epoch = 0
    current_step = 0
    finetune_steps = args.max_steps
    
    # dataset
    dataset_iterator, total_length = pretrain_dataset_provider.get_shard(epoch)
    logging.info(f"Epoch: {epoch + 1}")
    model.train()

    # Defining the profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(skip_first=10, wait=5, warmup=1, active=3),
        with_flops=True,
        on_trace_ready=trace_handler,
    ) as prof:

        prof.start()
        for batch_index_number, batch_index in enumerate(tqdm(dataset_iterator, smoothing=1)):
            if batch_index_number >= len(dataset_iterator) - 1:
                # skip last batch
                optimizer.zero_grad()
                continue
            
            # batch
            # (?: [32, 1], input_ids: [32, 128], attention_mask:[32, 128], token_type_ids (all 0): [32, 128], masked_lm_labels: [32, 128])
            batch = pretrain_dataset_provider.get_batch(batch_index)
            batch = tuple(t.to(device) for t in batch)  # Move to GPU
            
            # forward
            loss = model.forward(batch)
            
            # backward
            loss.backward()
            prof.step()
            
            # Update Optimizer
            optimizer.step()
            optimizer.zero_grad()
                
            current_step += 1
            last_lr = lr_scheduler.get_last_lr()[0]
            logging.info(f"step: {current_step}, batch index: {batch_index_number+1}, loss: {loss}, lr: {last_lr}")
            
            if current_step == finetune_steps:
                logging.info("end of finetuning")
                prof.stop()
                return


if __name__ == "__main__":
    main()
