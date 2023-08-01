import torch
import os
import logging
from tqdm import tqdm

from run_pretraining import parse_arguments
from pretraining.utils import count_parameters
from pretraining.stitch_utils import stitch
from pretraining.base import BasePretrainModel
from pretraining.dataset.pretraining_dataset import PreTrainingDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def sanity_modular():
    args = parse_arguments()
    
    # Load two pre-training model skeletons + supplied model config
    src_model1 = BasePretrainModel(args)
    src_model2 = BasePretrainModel(args)

    # checkpoint: OrderedDict with model params
    logging.info(f"Loading source model 1 from {args.src_model1_path}")
    checkpoint1 = torch.load(os.path.join(args.src_model1_path, "pytorch_model.bin"))
    src_model1.network.load_state_dict(checkpoint1)
    logging.info(f"# of params in src_model1: {count_parameters(src_model1.network)}")
    
    logging.info(f"Loading source model 2 from {args.src_model2_path}")
    checkpoint2 = torch.load(os.path.join(args.src_model2_path, "pytorch_model.bin"))
    src_model2.network.load_state_dict(checkpoint2)
    logging.info(f"# of params in src_model2: {count_parameters(src_model2.network)}")
    
    # define stitched model skeleton
    stitched_model = BasePretrainModel(args, model_type="stitched-bert-mlm")
    
    # stitch two source models
    logging.info("Stitching 2 models...")
    stitch(
        src_model1.network,
        src_model2.network,
        stitched_model.network,
        skip_layernorm_=args.skip_layernorm,
        extra_src_list=[],
    )
    logging.info(f"# of params in stitched_model: {count_parameters(stitched_model.network)}")
    
    # ===== stitch sanity check =====
    # embeddings
    assert torch.allclose(
        stitched_model.network.bert.embeddings.word_embeddings.weight,
        torch.concat([src_model1.network.bert.embeddings.word_embeddings.weight, src_model2.network.bert.embeddings.word_embeddings.weight], dim=1)
    ), "word embeddings do not match"
    
    # embeddings layer norm
    assert torch.allclose(
        stitched_model.network.bert.embeddings.LayerNorm.weight,
        torch.concat([src_model1.network.bert.embeddings.LayerNorm.weight, src_model2.network.bert.embeddings.LayerNorm.weight])
    ), "embedding layernorms do not match"
    
    # encoder layer 0
    assert torch.allclose(
        stitched_model.network.bert.encoder.layer[0].attention.self.query.weight,
        torch.stack([src_model1.network.bert.encoder.layer[0].attention.self.query.weight, src_model2.network.bert.encoder.layer[0].attention.self.query.weight])
    ), "encoder layer 0 query weights do not match"
    
    # encoder layer pre attention layer norm weight
    assert torch.allclose(
        stitched_model.network.bert.encoder.layer[0].PreAttentionLayerNorm.weight,
        torch.concat([src_model1.network.bert.encoder.layer[0].PreAttentionLayerNorm.weight, src_model2.network.bert.encoder.layer[0].PreAttentionLayerNorm.weight])
    ), "encoder layer 0 pre attention layer norm weights do not match"
    
    # encoder layer pre attention layer norm bias
    assert torch.allclose(
        stitched_model.network.bert.encoder.layer[0].PreAttentionLayerNorm.bias,
        torch.concat([src_model1.network.bert.encoder.layer[0].PreAttentionLayerNorm.bias, src_model2.network.bert.encoder.layer[0].PreAttentionLayerNorm.bias])
    ), "encoder layer 0 pre attention layer norm bias do not match"
    
    # blend layer
    assert torch.allclose(
        stitched_model.network.bert.encoder.layer[0].blend_layer.weight.data,
        torch.eye(stitched_model.network.config.hidden_size)
    ), "blend layer weights are not identity matrix"
    
    assert torch.allclose(
        stitched_model.network.bert.encoder.layer[0].blend_layer.bias.data,
        torch.zeros(stitched_model.network.config.hidden_size)
    ), "blend layer bias are not zero"
    
    src_model1.eval()
    src_model1.network.to(device)
    src_model2.eval()
    src_model2.network.to(device)
    stitched_model.eval()
    stitched_model.network.to(device)
    
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
        
        print(f"src_model1 loss: {src_model1.network(batch)[0].item()}")
        print(f"src_model2 loss: {src_model2.network(batch)[0].item()}")
        print(f"stitched_model loss: {stitched_model.network(batch)[0].item()}")

        # embedding output - [bsz, 128, 512]
        src_emb_out1 = src_model1.network.bert.embeddings(input_ids, token_type_ids, skip_ln_dp=True)
        src_emb_out2 = src_model2.network.bert.embeddings(input_ids, token_type_ids, skip_ln_dp=True)
        tgt_emb_out = stitched_model.network.bert.embeddings(input_ids, token_type_ids, skip_ln_dp=True)
        assert torch.allclose(torch.concat((src_emb_out1, src_emb_out2), dim=-1), tgt_emb_out, atol=1e-6)

        # self attn output - [bsz, 128, 512]
        src_attn_out1 = src_model1.network.bert.encoder.layer[0].attention(src_emb_out1, extended_attention_mask, skip_ln_dp=True)[0]
        src_attn_out2 = src_model2.network.bert.encoder.layer[0].attention(src_emb_out2, extended_attention_mask, skip_ln_dp=True)[0]
        tgt_attn_out = stitched_model.network.bert.encoder.layer[0].attention(tgt_emb_out, extended_attention_mask, skip_ln_dp=True)[0]
        assert torch.allclose(torch.concat((src_attn_out1, src_attn_out2), dim=-1), tgt_attn_out, atol=1e-6)
        
        # intermediate output - [bsz, 128, 2048]
        src_inter_out1 = src_model1.network.bert.encoder.layer[0].intermediate(src_attn_out1[0])
        src_inter_out2 = src_model2.network.bert.encoder.layer[0].intermediate(src_attn_out2[0])
        tgt_inter_out = stitched_model.network.bert.encoder.layer[0].intermediate(tgt_attn_out[0])
        assert torch.allclose(torch.concat((src_inter_out1, src_inter_out2), dim=-1), tgt_inter_out, atol=1e-6)
        
        # output output - [bsz, 128, 512]
        src_out_out1 = src_model1.network.bert.encoder.layer[0].output(src_inter_out1)
        src_out_out2 = src_model2.network.bert.encoder.layer[0].output(src_inter_out2)
        tgt_out_out = stitched_model.network.bert.encoder.layer[0].output(tgt_inter_out)
        assert torch.allclose(torch.concat((src_out_out1, src_out_out2), dim=-1), tgt_out_out, atol=1e-6)
        
        # bert layer output
        src_layer_out1 = src_model1.network.bert.encoder.layer[0](src_emb_out1, extended_attention_mask, skip_ln_dp=True)[0]
        src_layer_out2 = src_model2.network.bert.encoder.layer[0](src_emb_out2, extended_attention_mask, skip_ln_dp=True)[0]
        tgt_layer_out = stitched_model.network.bert.encoder.layer[0](tgt_emb_out, extended_attention_mask, skip_ln_dp=True)[0]
        assert torch.allclose(torch.concat((src_layer_out1, src_layer_out2), dim=-1), tgt_layer_out, atol=1e-6)
        
        breakpoint()


if __name__ == "__main__":
    sanity_modular()
