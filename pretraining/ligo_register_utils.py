import torch
from typing import List, Type
from torch.nn import ParameterList

from pretraining.modeling import (
    BertLMHeadModel,
    BertModel,
    BertOnlyMLMHead,
    BertEmbeddings,
    BertAttention,
    BertSelfAttention,
    BertLayer,
)

from pretraining.ligo_parameterization import (
    register_embedding,
    register_linear,
    register_ln,
)


def register_bert_embeddings(
    tgt_emb: Type[BertEmbeddings], src_emb_list: List[Type[BertEmbeddings]]
):
    # word embeddings
    register_embedding(
        tgt_emb=tgt_emb.word_embeddings,
        src_emb_list=[src_emb.word_embeddings for src_emb in src_emb_list],
    )

    # position embeddings
    # tie b to word embedding
    register_embedding(
        tgt_emb=tgt_emb.position_embeddings,
        src_emb_list=[src_emb.position_embeddings for src_emb in src_emb_list],
        tie_b=tgt_emb.word_embeddings.parametrizations.weight[0].ligo_b,
    )

    # token type embeddings
    # tie b to word embedding
    register_embedding(
        tgt_emb=tgt_emb.token_type_embeddings,
        src_emb_list=[src_emb.token_type_embeddings for src_emb in src_emb_list],
        tie_b=tgt_emb.word_embeddings.parametrizations.weight[0].ligo_b,
    )
    
    if tgt_emb.config.hf_architecture:
        register_ln(
            tgt_ln=tgt_emb.LayerNorm,
            src_ln_list=[src_emb.LayerNorm for src_emb in src_emb_list],
            tie_b=tgt_emb.word_embeddings.parametrizations.weight[0].ligo_b,
            bias=tgt_emb.LayerNorm.bias is not None,
        )


def register_self_attn(
    tgt_self_attn: Type[BertSelfAttention],
    src_self_attn_list: List[Type[BertSelfAttention]],
    b_emb=Type[ParameterList],
):
    # ligo_a of query, key, value are tied to b_emb
    # query
    register_linear(
        tgt_linear=tgt_self_attn.query,
        src_linear_list=[src_self_attn.query for src_self_attn in src_self_attn_list],
        tie_a=b_emb,
        tie_b=None,
        bias=tgt_self_attn.query.bias is not None,
    )

    # key
    register_linear(
        tgt_linear=tgt_self_attn.key,
        src_linear_list=[src_self_attn.key for src_self_attn in src_self_attn_list],
        tie_a=b_emb,
        tie_b=None,
        bias=tgt_self_attn.key.bias is not None,
    )

    # value
    register_linear(
        tgt_linear=tgt_self_attn.value,
        src_linear_list=[src_self_attn.value for src_self_attn in src_self_attn_list],
        tie_a=b_emb,
        tie_b=None,
        bias=tgt_self_attn.value.bias is not None,
    )


def register_attn(
    tgt_attn: Type[BertAttention],
    src_attn_list: List[Type[BertAttention]],
    b_emb=Type[ParameterList],
):
    # Key, query, value projections
    register_self_attn(
        tgt_self_attn=tgt_attn.self,
        src_self_attn_list=[src_attn.self for src_attn in src_attn_list],
        b_emb=b_emb,
    )

    # Output projection
    # ligo_a is tied to b_value, ligo_b is tied to b_emb
    register_linear(
        tgt_linear=tgt_attn.output.dense,
        src_linear_list=[src_attn.output.dense for src_attn in src_attn_list],
        tie_a=tgt_attn.self.value.parametrizations.weight[0].ligo_b,
        tie_b=b_emb,
        bias=tgt_attn.output.dense.bias is not None,
    )


def register_layer(
    tgt_layer: Type[BertLayer],
    src_layer_list: List[Type[BertLayer]],
    b_emb=Type[ParameterList],
):
    # Multihead attentions
    register_attn(
        tgt_attn=tgt_layer.attention,
        src_attn_list=[src_layer.attention for src_layer in src_layer_list],
        b_emb=b_emb,
    )

    # Intermediate ffn
    # ligo_a is tied to b_emb
    register_linear(
        tgt_linear=tgt_layer.intermediate.dense_act,
        src_linear_list=[src_layer.intermediate.dense_act for src_layer in src_layer_list],
        tie_a=b_emb,
        tie_b=None,
        bias=tgt_layer.intermediate.dense_act.bias is not None,
    )

    # Output ffn
    # ligo_a is tied to b_fc1, ligo_b is tied to b_emb
    register_linear(
        tgt_linear=tgt_layer.output.dense,
        src_linear_list=[src_layer.output.dense for src_layer in src_layer_list],
        tie_a=tgt_layer.intermediate.dense_act.parametrizations.weight[0].ligo_b,
        tie_b=b_emb,
        bias=tgt_layer.output.dense.bias is not None,
    )

    # copy both PreAttentionLayerNorm, PostAttentionLayerNorm
    register_ln(
        tgt_ln=tgt_layer.PreAttentionLayerNorm,
        src_ln_list=[src_layer.PreAttentionLayerNorm for src_layer in src_layer_list],
        tie_b=b_emb,
        bias=tgt_layer.PreAttentionLayerNorm.bias is not None,
    )
    register_ln(
        tgt_ln=tgt_layer.PostAttentionLayerNorm,
        src_ln_list=[src_layer.PostAttentionLayerNorm for src_layer in src_layer_list],
        tie_b=b_emb,
        bias=tgt_layer.PostAttentionLayerNorm.bias is not None,
    )


def register_bert(tgt_bert: Type[BertModel], src_bert_list: List[Type[BertModel]]):
    # ==== register embeddings
    register_bert_embeddings(
        tgt_emb=tgt_bert.embeddings,
        src_emb_list=[src_model.embeddings for src_model in src_bert_list],
    )

    # This param will be shared multiple times
    b_emb = tgt_bert.embeddings.word_embeddings.parametrizations.weight[0].ligo_b

    # ===== register encoder layers
    assert len(tgt_bert.encoder.layer) == len(src_bert_list[0].encoder.layer)
    n_layers = len(tgt_bert.encoder.layer)
    for l in range(n_layers):
        register_layer(
            tgt_bert.encoder.layer[l],
            [src_bert_list[i].encoder.layer[l] for i in range(len(src_bert_list))],
            b_emb
        )

    # when using hf_architecture, no final LayerNorm and Pooler
    if not tgt_bert.config.hf_architecture:
        # ===== register final LayerNorm
        register_ln(
            tgt_ln=tgt_bert.encoder.FinalLayerNorm,
            src_ln_list=[src_bert.encoder.FinalLayerNorm for src_bert in src_bert_list],
            tie_b=b_emb,
            bias=tgt_bert.encoder.FinalLayerNorm.bias is not None,
        )

        # NOTE: pooler is not used during pretraining
        # ===== register pooler
        register_linear(
            tgt_linear=tgt_bert.pooler.dense_act,
            src_linear_list=[src_bert.pooler.dense_act for src_bert in src_bert_list],
            # TODO: check
            tie_a=None,
            tie_b=None,
            bias=tgt_bert.pooler.dense_act.bias is not None,
        )


def register_mlm_head(
    tgt_mlm_head: Type[BertOnlyMLMHead],
    src_mlm_head_list: List[Type[BertOnlyMLMHead]],
    b_emb=Type[ParameterList],
):
    # register linear in BertPredictionHeadTransform
    # TODO: check if we should share b_emb or learn new ligos
    # Now tie only ligo_a to b_emb
    register_linear(
        tgt_linear=tgt_mlm_head.predictions.transform.dense_act,
        src_linear_list=[
            src_mlm_head.predictions.transform.dense_act for src_mlm_head in src_mlm_head_list
        ],
        tie_a=b_emb,
        tie_b=None,
        bias=tgt_mlm_head.predictions.transform.dense_act.bias is not None,
    )

    # register LN in BertPredictionHeadTransform
    # tie ligo_b to transform.dense_act.ligo_b
    register_ln(
        tgt_ln=tgt_mlm_head.predictions.transform.LayerNorm,
        src_ln_list=[
            src_mlm_head.predictions.transform.LayerNorm for src_mlm_head in src_mlm_head_list
        ],
        tie_b=tgt_mlm_head.predictions.transform.dense_act.parametrizations.weight[0].ligo_b,
        bias=tgt_mlm_head.predictions.transform.LayerNorm.bias is not None,
    )

    # register decoder
    # TODO: this should be similar to embedding, no bias (but bias exist in hf)
    # (decoder): Linear(in_features=1024, out_features=30528, bias=False)
    # TODO: what is cls.predictions.bias?
    register_linear(
        tgt_linear=tgt_mlm_head.predictions.decoder,
        src_linear_list=[src_mlm_head.predictions.decoder for src_mlm_head in src_mlm_head_list],
        tie_a=b_emb,
        tie_b=None,
        bias=tgt_mlm_head.predictions.decoder.bias is not None,
        is_decoder=True,
    )


def register_models(tgt_model: Type[BertLMHeadModel], src_model_list: Type[BertLMHeadModel]):
    # register BertModel
    register_bert(
        tgt_bert=tgt_model.bert,
        src_bert_list=[src_model.bert for src_model in src_model_list],
    )

    # register BertOnlyMLMHead
    register_mlm_head(
        tgt_mlm_head=tgt_model.cls,
        src_mlm_head_list=[src_model.cls for src_model in src_model_list],
        b_emb=tgt_model.bert.embeddings.word_embeddings.parametrizations.weight[0].ligo_b,
    )


# ====================
# functions for sanity check
# ====================
def check_tied_weights(tgt_model):
    state_dict = tgt_model.state_dict()
    
    def _get_identical_params(param, state_dict):
        tied_param_list = []
        for n, p in state_dict.items():
            if param.shape != p.shape:
                continue
            
            if torch.isclose(param, p).all():
                tied_param_list.append(n)
        return tied_param_list
    
    # check b_emb
    b_emb_0 = state_dict['bert.embeddings.word_embeddings.parametrizations.weight.0.ligo_b.0']
    b_emb_1 = state_dict['bert.embeddings.word_embeddings.parametrizations.weight.0.ligo_b.1']
    
    b_emb_0_tied_list = _get_identical_params(b_emb_0, state_dict)
    b_emb_1_tied_list = _get_identical_params(b_emb_1, state_dict)
    
    # check b_value of the first layer
    b_value = state_dict['bert.encoder.layer.0.attention.self.value.parametrizations.weight.0.ligo_b.0']
    b_value_tied_list = _get_identical_params(b_value, state_dict)
    
    # check b_fc1
    b_fc1 = state_dict['bert.encoder.layer.0.intermediate.dense_act.parametrizations.weight.0.ligo_b.0']
    b_fc1_tied_list = _get_identical_params(b_fc1, state_dict)
    
    breakpoint()
