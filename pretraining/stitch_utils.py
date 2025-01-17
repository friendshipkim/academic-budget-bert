"""
util functions to stitch two BertLMHeadModel models
"""
import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm
from typing import Type, Union, List

from pretraining.modeling import (
    BertLMHeadModel,
    BertModel,
    BertOnlyMLMHead,
    BertEmbeddings,
    BertAttention,
    BertSelfAttention,
    BertLayer,
    LinearActivation,
)
from pretraining.configs import PretrainedBertConfig


# global vars
epsilon = 0
skip_layernorm = False
stitch4 = False


# TODO: merge this into StitchedBertConfig
def check_if_stitchable(
    src1_cfg: Type[PretrainedBertConfig],
    src2_cfg: Union[Type[PretrainedBertConfig], None],
) -> None:
    """
    Given two bert configs, check if the two models are stitchable
    Args:
        src1_cfg (PretrainedBertConfig): first source model config
        src2_cfg (PretrainedBertConfig or None): second source model config
    """
    # if src2 model is None, return
    if src2_cfg is None:
        return

    assert src1_cfg.vocab_size == src2_cfg.vocab_size, "vocab sizes should match"
    assert (
        src1_cfg.num_hidden_layers == src2_cfg.num_hidden_layers
    ), "number of hidden layers should match"
    assert (
        src1_cfg.hidden_size / src1_cfg.num_attention_heads
        == src2_cfg.hidden_size / src2_cfg.num_attention_heads
    ), "attention head size should match"


def copy_linear(
    src1: Union[Type[nn.Linear], Type[LinearActivation]],
    src2: Union[Type[nn.Linear], Type[LinearActivation]],
    tgt: Union[Type[nn.Linear], Type[LinearActivation]],
    extra_src_list: List[Union[Type[nn.Linear], Type[LinearActivation], None]],
) -> None:
    """
    Diagonally copy the weights of the two source Linear layers to the target layer.
    Set non-diagonal parts to epsilon
    Args:
        src1 (torch.nn.Linear or LinearActivation): first source Linear layer
        src2 (torch.nn.Linear or LinearActivation): second source Linear layer
        tgt (torch.nn.Linear or LinearActivation): target Linear layer
        extra_src_list (List[torch.nn.Linear or LinearActivation]): third, fourth source Linear layer if needed
    """
    if stitch4:
        src3, src4 = extra_src_list[0], extra_src_list[1]

        # Check if bias exists
        assert None not in (
            src1.bias,
            src2.bias,
            src3.bias,
            src4.bias,
            tgt.bias,
        ) or not any((src1.bias, src2.bias, src3.bias, src4.bias, tgt.bias))

        src1_out_dim, src1_in_dim = src1.weight.size()
        src2_out_dim, src2_in_dim = src2.weight.size()
        src3_out_dim, src3_in_dim = src3.weight.size()
        src4_out_dim, src4_in_dim = src4.weight.size()
        tgt_out_dim, tgt_in_dim = tgt.weight.size()

        # check dimensions
        assert tgt_out_dim == src1_out_dim + src2_out_dim
        assert tgt_in_dim == src1_in_dim + src3_in_dim

        # Copy weights
        tgt.weight.data[:src1_out_dim, :src1_in_dim] = src1.weight.data
        tgt.weight.data[-src2_out_dim:, :src2_in_dim] = src2.weight.data
        tgt.weight.data[:src3_out_dim, -src3_in_dim:] = src3.weight.data
        tgt.weight.data[-src4_out_dim:, -src4_in_dim:] = src4.weight.data

        # If biases exist, copy biases
        if tgt.bias is not None:
            tgt.bias.data[:src1_out_dim] = (src1.bias.data + src2.bias.data) / 2
            tgt.bias.data[-src3_out_dim:] = (src3.bias.data + src4.bias.data) / 2

    else:
        # Check if bias exists
        assert None not in (src1.bias, src2.bias, tgt.bias) or not any(
            (src1.bias, src2.bias, tgt.bias)
        )

        src1_out_dim, src1_in_dim = src1.weight.size()
        src2_out_dim, src2_in_dim = src2.weight.size()
        tgt_out_dim, tgt_in_dim = tgt.weight.size()

        assert tgt_out_dim == src1_out_dim + src2_out_dim
        assert tgt_in_dim == src1_in_dim + src2_in_dim

        # Initialize with epsilon
        tgt.weight.data[:] = epsilon

        # # initialize to normal dist
        # mu, std = 0, 1e-3
        # tgt.weight.data[:] = torch.randn_like(tgt.weight.data) * std + mu

        # Copy weights diagonally
        tgt.weight.data[:src1_out_dim, :src1_in_dim] = src1.weight.data
        tgt.weight.data[-src2_out_dim:, -src2_in_dim:] = src2.weight.data

        # If biases exist, copy biases
        if tgt.bias is not None:
            tgt.bias.data[:src1_out_dim] = src1.bias.data
            tgt.bias.data[-src2_out_dim:] = src2.bias.data


def copy_layernorm(
    src1: Union[Type[nn.LayerNorm], Type[FusedLayerNorm]],
    src2: Union[Type[nn.LayerNorm], Type[FusedLayerNorm]],
    tgt: Union[Type[nn.LayerNorm], Type[FusedLayerNorm]],
    extra_src_list: List[Union[Type[nn.LayerNorm], Type[FusedLayerNorm], None]],
) -> None:
    """
    Copy the weights of the two source LayerNorm layers to the target layer
    Args:
        src1 (torch.nn.LayerNorm or apex FusedLayerNorm): first source LayerNorm
        src2 (torch.nn.LayerNorm or apex FusedLayerNorm): second source LayerNorm
        tgt (torch.nn.LayerNorm or apex FusedLayerNorm): target LayerNorm
        extra_src_list (List[torch.nn.LayerNorm or apex FusedLayerNorm]): third, fourth source LayerNorm if needed
    """
    if stitch4:
        src3, src4 = extra_src_list[0], extra_src_list[1]
        src1_dim, src2_dim, src3_dim, src4_dim, tgt_dim = (
            src1.weight.size(0),
            src2.weight.size(0),
            src3.weight.size(0),
            src4.weight.size(0),
            tgt.weight.size(0),
        )
        assert src1_dim == src2_dim
        assert src3_dim == src4_dim
        assert tgt_dim == src1_dim + src3_dim

        # Copy weights
        # NOTE: if stitching two different models: (src1.weight.data + src2.weight.data) / 2
        tgt.weight.data[:src1_dim] = (src1.weight.data + src2.weight.data) / 2
        tgt.weight.data[-src3_dim:] = (src3.weight.data + src4.weight.data) / 2

        # Copy biases
        tgt.bias.data[:src1_dim] = (src1.bias.data + src2.bias.data) / 2
        tgt.bias.data[-src3_dim:] = (src3.bias.data + src4.bias.data) / 2

    else:
        src1_dim, src2_dim, tgt_dim = (
            src1.weight.size(0),
            src2.weight.size(0),
            tgt.weight.size(0),
        )
        assert tgt_dim == src1_dim + src2_dim

        # Copy weights
        # NOTE: if stitching two different models: (src1.weight.data + src2.weight.data) / 2
        tgt.weight.data[:src1_dim] = src1.weight.data  # / 2
        tgt.weight.data[-src2_dim:] = src2.weight.data  # / 2

        # Copy biases
        tgt.bias.data[:src1_dim] = src1.bias.data  # / 2
        tgt.bias.data[-src2_dim:] = src2.bias.data  # / 2


def copy_self_attn(
    src1: Type[BertSelfAttention],
    src2: Type[BertSelfAttention],
    tgt: Type[BertSelfAttention],
    extra_src_list: List[Union[Type[BertSelfAttention], None]],
) -> None:
    """
    Copy the linear projections of the two source BertSelfAttention modules to the target module
    Set the rest to epsilon
    Args:
        src1 (BertSelfAttention): first source BertSelfAttention module
        src2 (BertSelfAttention): second source BertSelfAttention module
        tgt (BertSelfAttention): target BertSelfAttention module
        extra_src_list (list): third, fourth source BertSelfAttentions if needed
    """
    # copy linear layers of query, key, value
    copy_linear(
        src1.query,
        src2.query,
        tgt.query,
        extra_src_list=[src.query for src in extra_src_list] if stitch4 else [],
    )
    copy_linear(
        src1.key,
        src2.key,
        tgt.key,
        extra_src_list=[src.key for src in extra_src_list] if stitch4 else [],
    )
    copy_linear(
        src1.value,
        src2.value,
        tgt.value,
        extra_src_list=[src.value for src in extra_src_list] if stitch4 else [],
    )


def copy_attention(
    src1: Type[BertAttention],
    src2: Type[BertAttention],
    tgt: Type[BertAttention],
    extra_src_list: List[Union[Type[BertAttention], None]],
) -> None:
    """
    Copy input/output linear projections and layernorm of the two source BertAttention modules to the target module
    Set the rest to epsilon
    Args:
        src1 (BertAttention): first source BertAttention module
        src2 (BertAttention): second source BertAttention module
        tgt (BertAttention): target BertAttention module
        extra_src_list (list): third, fourth source BertAttentions if needed
    """
    # Key, query, value projections
    copy_self_attn(
        src1.self,
        src2.self,
        tgt.self,
        extra_src_list=[src.self for src in extra_src_list] if stitch4 else [],
    )

    # Output projection
    copy_linear(
        src1.output.dense,
        src2.output.dense,
        tgt.output.dense,
        extra_src_list=[src.output.dense for src in extra_src_list] if stitch4 else [],
    )

    # # Layernorm
    # if not skip_layernorm:
    #     copy_layernorm(src1.output.LayerNorm, src2.output.LayerNorm, tgt.output.LayerNorm)


def copy_layer(
    src1: Type[BertLayer],
    src2: Type[BertLayer],
    tgt: Type[BertLayer],
    extra_src_list: List[Union[Type[BertLayer], None]],
) -> None:
    """
    Copy "" of the two source Bert layers to the target layer
    Args:
        src1 (transformers.models.bert.modeling_bert.BertLayer): first source BertLayer
        src2 (transformers.models.bert.modeling_bert.BertLayer): second source BertLayer
        tgt (transformers.models.bert.modeling_bert.BertLayer): target BertLayer
        extra_src_list (list): third, fourth source BertLayers if needed
    """
    # Multihead attentions
    copy_attention(
        src1.attention,
        src2.attention,
        tgt.attention,
        extra_src_list=[src.attention for src in extra_src_list] if stitch4 else [],
    )

    # Intermediate ffn
    copy_linear(
        src1.intermediate.dense_act,
        src2.intermediate.dense_act,
        tgt.intermediate.dense_act,
        extra_src_list=[src.intermediate.dense_act for src in extra_src_list] if stitch4 else [],
    )

    # Output ffn
    copy_linear(
        src1.output.dense,
        src2.output.dense,
        tgt.output.dense,
        extra_src_list=[src.output.dense for src in extra_src_list] if stitch4 else [],
    )
    # # NOTE: No output layernorm
    # if not skip_layernorm:
    #     copy_layernorm(src1.output.LayerNorm, src2.output.LayerNorm, tgt.output.LayerNorm)

    # copy both PreAttentionLayerNorm, PostAttentionLayerNorm
    if not skip_layernorm:
        copy_layernorm(
            src1.PreAttentionLayerNorm,
            src2.PreAttentionLayerNorm,
            tgt.PreAttentionLayerNorm,
            extra_src_list=[src.PreAttentionLayerNorm for src in extra_src_list] if stitch4 else [],
        )
        copy_layernorm(
            src1.PostAttentionLayerNorm,
            src2.PostAttentionLayerNorm,
            tgt.PostAttentionLayerNorm,
            extra_src_list=[src.PostAttentionLayerNorm for src in extra_src_list] if stitch4 else [],
        )


def copy_embeddings(
    src1: Type[BertEmbeddings],
    src2: Type[BertEmbeddings],
    tgt: Type[BertEmbeddings],
    extra_src_list: List[Union[Type[BertEmbeddings], None]],
) -> None:
    """
    Copy embeddings and layernorm of the two source BertEmbeddings modules to the target module
    Args:
        src1 (BertEmbeddings): first source BertEmbeddings module
        src2 (BertEmbeddings): second source BertEmbeddings module
        tgt (BertEmbeddings): target BertEmbeddings module
        extra_src_list (list): third, fourth source BertEmbeddings if needed
    """
    # Embeddings
    embed_types = ["word_embeddings", "position_embeddings", "token_type_embeddings"]

    if stitch4:
        src3, src4 = extra_src_list
        for embed_type in embed_types:
            tgt.get_submodule(embed_type).weight.data[:] = torch.cat(
                (
                    (src1.get_submodule(embed_type).weight.data
                     + src2.get_submodule(embed_type).weight.data) / 2,
                    (src3.get_submodule(embed_type).weight.data
                     + src4.get_submodule(embed_type).weight.data) / 2,
                ),
                dim=-1,
            )
    else:
        for embed_type in embed_types:
            tgt.get_submodule(embed_type).weight.data[:] = torch.cat(
                (
                    src1.get_submodule(embed_type).weight.data,
                    src2.get_submodule(embed_type).weight.data,
                ),
                dim=-1,
            )

    # # Embedding layernorm
    # if not skip_layernorm:
    #     copy_layernorm(src1.LayerNorm, src2.LayerNorm, tgt.LayerNorm)


def copy_bert(
    src1: Type[BertModel],
    src2: Type[BertModel],
    tgt: Type[BertModel],
    extra_src_list: List[Type[BertModel]],
) -> None:
    """Copy two source BertModels to the target BertModel
    Args:
        src1 (BertModel): first source BertModel
        src2 (BertModel): second source BertModel
        tgt (BertModel): target BertModel
        extra_src_list (list): third, fourth source BertModels if needed
    """
    # Embeddings
    copy_embeddings(
        src1.embeddings,
        src2.embeddings,
        tgt.embeddings,
        extra_src_list=[src.embeddings for src in extra_src_list] if stitch4 else [])

    # Copy transformer layers
    n_layers = len(src1.encoder.layer)
    if stitch4:
        extra_layer_list = [
            [layer_3, layer_4]
            for layer_3, layer_4 in zip(
                extra_src_list[0].encoder.layer, extra_src_list[1].encoder.layer
            )
        ]
    else:
        extra_layer_list = [[] for _ in range(n_layers)]

    for layer_1, layer_2, layer_st, extra_layers in zip(
        src1.encoder.layer, src2.encoder.layer, tgt.encoder.layer, extra_layer_list
    ):
        copy_layer(layer_1, layer_2, layer_st, extra_layers)

    # NOTE: copy final LayerNorm
    if not skip_layernorm:
        copy_layernorm(
            src1.encoder.FinalLayerNorm,
            src2.encoder.FinalLayerNorm,
            tgt.encoder.FinalLayerNorm,
            extra_src_list=[src.encoder.FinalLayerNorm for src in extra_src_list] if stitch4 else [],
        )

    # Pooler
    copy_linear(
        src1.pooler.dense_act,
        src2.pooler.dense_act,
        tgt.pooler.dense_act,
        extra_src_list=[src.pooler.dense_act for src in extra_src_list] if stitch4 else [],
    )


def copy_mlm_head(
    src1: Type[BertOnlyMLMHead],
    src2: Type[BertOnlyMLMHead],
    tgt: Type[BertOnlyMLMHead],
    extra_src_list: List[Type[BertOnlyMLMHead]],
) -> None:
    """Copy two source BertOnlyMLMHead to the target BertOnlyMLMHead

    Args:
        src1 (BertOnlyMLMHead): first source BertOnlyMLMHead
        src2 (BertOnlyMLMHead): second source BertOnlyMLMHead
        tgt (BertOnlyMLMHead): target BertOnlyMLMHead
        extra_src_list (list): third, fourth source BertOnlyMLMHead if needed
    """
    
    # copy BertPredictionHeadTransform
    copy_linear(
        src1.predictions.transform.dense_act,
        src2.predictions.transform.dense_act,
        tgt.predictions.transform.dense_act,
        extra_src_list=[src.predictions.transform.dense_act for src in extra_src_list] if stitch4 else [],
    )
    if not skip_layernorm:
        copy_layernorm(
            src1.predictions.transform.LayerNorm,
            src2.predictions.transform.LayerNorm,
            tgt.predictions.transform.LayerNorm,
            extra_src_list=[src.predictions.transform.LayerNorm for src in extra_src_list] if stitch4 else [],
        )

    # copy decoder of BertLMPredictionHead
    # concat along the last axis, size: hidden_size x vocab_size
    if stitch4:
        src3, src4 = extra_src_list[0], extra_src_list[1]
        tgt.predictions.decoder.weight.data[:] = torch.cat(
            (
                (src1.predictions.decoder.weight.data + src2.predictions.decoder.weight.data) / 2,
                (src3.predictions.decoder.weight.data + src4.predictions.decoder.weight.data) / 2,
            ),
            dim=-1,
        )
    else:
        tgt.predictions.decoder.weight.data[:] = torch.cat(
            (
                src1.predictions.decoder.weight.data,
                src2.predictions.decoder.weight.data,
            ),
            dim=-1,
        )


def make_dummy_model(
    src1: Type[BertLMHeadModel], tgt: Type[BertLMHeadModel]
):
    """
    Make dummy model to which can be stitched with src1 to make tgt
    Args:
        src1 (BertLMHeadModel): source model to stitch
        tgt (BertLMHeadModel): stitched target model
    """
    dummy_cfg = PretrainedBertConfig(**src1.config.to_dict())

    # update config
    dummy_cfg._name_or_path = "dummy"
    dummy_cfg.hidden_size = tgt.config.hidden_size - src1.config.hidden_size
    dummy_cfg.intermediate_size = dummy_cfg.hidden_size * 4

    attn_head_size = src1.config.hidden_size // src1.config.num_attention_heads
    dummy_cfg.num_attention_heads = dummy_cfg.hidden_size // attn_head_size

    # define dummy model
    dummy_model = src1.__class__(dummy_cfg)

    # update all parameters with epsilon
    for p in dummy_model.parameters():
        p.data[:] = epsilon

    return dummy_model


def stitch(
    src1: Type[BertLMHeadModel],
    src2: Type[BertLMHeadModel],
    tgt: Type[BertLMHeadModel],
    skip_layernorm_flg: bool,
    extra_src_list: List[Type[BertLMHeadModel]],
) -> None:
    """
    Stitch two Bert models by copying the internal weights
    Args:
        src1 (BertLMHeadModel): first source model to stitch
        src2 (BertLMHeadModel or None): second source model to stitch, if None, only copy the first model
        tgt (BertLMHeadModel): stitched target model
        skip_layernorm_flg (bool): whether not to stitch layernorms
        extra_src_list (list): third, fourth source models if needed
    """
    global epsilon, skip_layernorm, stitch4
    
    # overwrite global vars
    epsilon = tgt.config.epsilon
    skip_layernorm = skip_layernorm_flg
    stitch_4 = len(extra_src_list) != 0

    # if only one source model is given, make dummy model with epsilon
    if src2 is None:
        src2 = make_dummy_model(src1, tgt, epsilon)

    # check if two models are stitchable
    check_if_stitchable(src1.config, src2.config)

    # copy BertModel
    copy_bert(
        src1.bert,
        src2.bert,
        tgt.bert,
        extra_src_list=[src.bert for src in extra_src_list] if stitch_4 else [],
    )

    # copy BertOnlyMLMHead
    copy_mlm_head(
        src1.cls,
        src2.cls,
        tgt.cls,
        extra_src_list=[src.cls for src in extra_src_list] if stitch_4 else [],
    )
