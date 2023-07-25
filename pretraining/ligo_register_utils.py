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
    register_decoder_linear,
    register_ln,
    set_init_type,
)


# global vars
tie_flag = True
inherit_flag = False
skip_ln = False


def register_bert_embeddings(
    tgt_emb: Type[BertEmbeddings], src_emb_list: List[Type[BertEmbeddings]]
):
    print("=== register embeddings ===")
    # word embeddings
    print("- word embeddings")
    register_embedding(
        tgt_emb=tgt_emb.word_embeddings,
        src_emb_list=[src_emb.word_embeddings for src_emb in src_emb_list],
    )
    b_emb = tgt_emb.word_embeddings.parametrizations.weight[0].ligo_b
    
    # position embeddings
    # tie b to word embedding
    print("- position embeddings")
    register_embedding(
        tgt_emb=tgt_emb.position_embeddings,
        src_emb_list=[src_emb.position_embeddings for src_emb in src_emb_list],
        tie_b=b_emb if tie_flag else None,
    )

    # token type embeddings
    # tie b to word embedding
    print("- token type embeddings")
    register_embedding(
        tgt_emb=tgt_emb.token_type_embeddings,
        src_emb_list=[src_emb.token_type_embeddings for src_emb in src_emb_list],
        tie_b=b_emb if tie_flag else None,
    )
    
    # layernorm
    if tgt_emb.config.hf_architecture and not skip_ln:
        print("- embedding ln")
        register_ln(
            tgt_ln=tgt_emb.LayerNorm,
            src_ln_list=[src_emb.LayerNorm for src_emb in src_emb_list],
            tie_b=b_emb if tie_flag else None,
            bias=tgt_emb.LayerNorm.bias is not None,
            init_b=[b_emb[i].detach() for i in range(len(b_emb))] if inherit_flag else None,
        )
    else:
        print("- skipping embedding ln")


def register_self_attn(
    tgt_self_attn: Type[BertSelfAttention],
    src_self_attn_list: List[Type[BertSelfAttention]],
    b_emb=Type[ParameterList],
    a_init_from_prev=List[Type[torch.Tensor]],
):
    print("=== register self attention ===")
    # ligo_a of query, key, value are tied to b_emb
    # query
    print("- query")
    register_linear(
        tgt_linear=tgt_self_attn.query,
        src_linear_list=[src_self_attn.query for src_self_attn in src_self_attn_list],
        tie_a=b_emb if tie_flag else None,
        tie_b=None,
        bias=tgt_self_attn.query.bias is not None,
        init_a=a_init_from_prev if inherit_flag else None,
    )
    
    # key
    print("- key")
    register_linear(
        tgt_linear=tgt_self_attn.key,
        src_linear_list=[src_self_attn.key for src_self_attn in src_self_attn_list],
        tie_a=b_emb if tie_flag else None,
        tie_b=None,
        bias=tgt_self_attn.key.bias is not None,
        init_a=a_init_from_prev if inherit_flag else None,
    )

    # value
    print("- value")
    register_linear(
        tgt_linear=tgt_self_attn.value,
        src_linear_list=[src_self_attn.value for src_self_attn in src_self_attn_list],
        tie_a=b_emb if tie_flag else None,
        tie_b=None,
        bias=tgt_self_attn.value.bias is not None,
        init_a=a_init_from_prev if inherit_flag else None,
    )


def register_attn(
    tgt_attn: Type[BertAttention],
    src_attn_list: List[Type[BertAttention]],
    b_emb=Type[ParameterList],
    a_init_from_prev=List[Type[torch.Tensor]],
):
    print("=== register attention ===")
    # Key, query, value projections
    register_self_attn(
        tgt_self_attn=tgt_attn.self,
        src_self_attn_list=[src_attn.self for src_attn in src_attn_list],
        b_emb=b_emb,
        a_init_from_prev=a_init_from_prev,
    )

    # Output projection
    # ligo_a is tied to b_value, ligo_b is tied to b_emb
    print("- output projection")
    register_linear(
        tgt_linear=tgt_attn.output.dense,
        src_linear_list=[src_attn.output.dense for src_attn in src_attn_list],
        tie_a=tgt_attn.self.value.parametrizations.weight[0].ligo_b if tie_flag else None,
        tie_b=b_emb if tie_flag else None,
        bias=tgt_attn.output.dense.bias is not None,
        init_a=[getattr(tgt_attn.self.value.parametrizations.weight[0], f"e_inv_{i}") for i in range(len(src_attn_list))] if inherit_flag else None,
    )


def register_layer(
    tgt_layer: Type[BertLayer],
    src_layer_list: List[Type[BertLayer]],
    b_emb=Type[ParameterList],
    a_init_from_prev=List[Type[torch.Tensor]],
):
    # Multihead attentions
    register_attn(
        tgt_attn=tgt_layer.attention,
        src_attn_list=[src_layer.attention for src_layer in src_layer_list],
        b_emb=b_emb,
        a_init_from_prev=a_init_from_prev,
    )

    # Intermediate ffn
    # ligo_a is tied to b_emb
    print("- intermediate ffn")
    if hasattr(tgt_layer.attention.output.dense.parametrizations.weight[0], "e_inv_0"):
        # pca untied
        tie_a = None
        init_a = [getattr(tgt_layer.attention.output.dense.parametrizations.weight[0], f"e_inv_{i}") for i in range(len(src_layer_list))]
    else:
        tie_a = tgt_layer.attention.output.dense.parametrizations.weight[0].ligo_b if tie_flag else None
        init_a = None
    
    register_linear(
        tgt_linear=tgt_layer.intermediate.dense_act,
        src_linear_list=[src_layer.intermediate.dense_act for src_layer in src_layer_list],
        tie_a=tie_a,
        tie_b=None,
        bias=tgt_layer.intermediate.dense_act.bias is not None,
        init_a=init_a,
    )

    # Output ffn
    # ligo_a is tied to b_fc1, ligo_b is tied to b_emb
    print("- output ffn")
    if hasattr(tgt_layer.intermediate.dense_act.parametrizations.weight[0], "e_inv_0"):
        # pca untied
        tie_a = None
        init_a = [getattr(tgt_layer.intermediate.dense_act.parametrizations.weight[0], f"e_inv_{i}") for i in range(len(src_layer_list))]
    else:
        tie_a = tgt_layer.intermediate.dense_act.parametrizations.weight[0].ligo_b if tie_flag else None
        init_a = None
    
    register_linear(
        tgt_linear=tgt_layer.output.dense,
        src_linear_list=[src_layer.output.dense for src_layer in src_layer_list],
        tie_a=tie_a,
        tie_b=b_emb if tie_flag else None,
        bias=tgt_layer.output.dense.bias is not None,
        init_a=init_a,
    )

    # copy both PreAttentionLayerNorm, PostAttentionLayerNorm
    # NOTE: ln was always tied before
    if not skip_ln:
        print("- layer norm")
        prev_ligo_b = tgt_layer.attention.output.dense.parametrizations.weight[0].ligo_b
        register_ln(
            tgt_ln=tgt_layer.PreAttentionLayerNorm,
            src_ln_list=[src_layer.PreAttentionLayerNorm for src_layer in src_layer_list],
            tie_b=prev_ligo_b if tie_flag else None,
            bias=tgt_layer.PreAttentionLayerNorm.bias is not None,
            init_b=[prev_ligo_b[i].detach() for i in range(len(prev_ligo_b))] if inherit_flag else None,
        )
        
        prev_ligo_b = tgt_layer.output.dense.parametrizations.weight[0].ligo_b
        register_ln(
            tgt_ln=tgt_layer.PostAttentionLayerNorm,
            src_ln_list=[src_layer.PostAttentionLayerNorm for src_layer in src_layer_list],
            tie_b=prev_ligo_b if tie_flag else None,
            bias=tgt_layer.PostAttentionLayerNorm.bias is not None,
            init_b=[prev_ligo_b[i].detach() for i in range(len(prev_ligo_b))] if inherit_flag else None,
        )
    else:
        print("- skipping layer norm")


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
    n_src_models = len(src_bert_list)
    
    for l in range(n_layers):
        print(f"=== register layer {l} ===")
        # if inherit from previous layer, use previous layer's e_inv as init
        if inherit_flag and not tie_flag:
            prev_layer = tgt_bert.embeddings.word_embeddings if l == 0 else tgt_bert.encoder.layer[l-1].output.dense
            a_init_from_prev = [getattr(prev_layer.parametrizations.weight[0], f"e_inv_{i}") for i in range(n_src_models)]
        else:
            a_init_from_prev = None
        
        register_layer(
            tgt_layer=tgt_bert.encoder.layer[l],
            src_layer_list=[src_bert_list[i].encoder.layer[l] for i in range(len(src_bert_list))],
            b_emb=b_emb,
            a_init_from_prev=a_init_from_prev
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
    a_init_from_prev=List[Type[torch.Tensor]],
):
    print("=== register mlm head ===")
    # register linear in BertPredictionHeadTransform
    # TODO: check if we should share b_emb or learn new ligos
    # Now tie only ligo_a to b_emb
    print("- predictions linear")
    register_linear(
        tgt_linear=tgt_mlm_head.predictions.transform.dense_act,
        src_linear_list=[
            src_mlm_head.predictions.transform.dense_act for src_mlm_head in src_mlm_head_list
        ],
        tie_a=b_emb if tie_flag else None,
        tie_b=None,
        bias=tgt_mlm_head.predictions.transform.dense_act.bias is not None,
        init_a=a_init_from_prev if inherit_flag else None,
    )
    
    # register LN in BertPredictionHeadTransform
    # tie ligo_b to transform.dense_act.ligo_b
    print("- predictions ln")
    if not skip_ln:
        prev_ligo_b = tgt_mlm_head.predictions.transform.dense_act.parametrizations.weight[0].ligo_b
        register_ln(
            tgt_ln=tgt_mlm_head.predictions.transform.LayerNorm,
            src_ln_list=[
                src_mlm_head.predictions.transform.LayerNorm for src_mlm_head in src_mlm_head_list
            ],
            tie_b=prev_ligo_b if tie_flag else None,
            bias=tgt_mlm_head.predictions.transform.LayerNorm.bias is not None,
            init_b=[prev_ligo_b[i].detach() for i in range(len(prev_ligo_b))] if inherit_flag else None,
        )
    else:
        print("- skipping predictions ln")

    # register decoder
    # TODO: this should be similar to embedding, no bias (but bias exist in hf)
    # (decoder): Linear(in_features=1024, out_features=30528, bias=False)
    # TODO: what is cls.predictions.bias?
    print("- decoder")
    register_decoder_linear(
        tgt_linear=tgt_mlm_head.predictions.decoder,
        src_linear_list=[src_mlm_head.predictions.decoder for src_mlm_head in src_mlm_head_list],
        tie_a=b_emb,
        bias=tgt_mlm_head.predictions.decoder.bias is not None,
    )


def register_models(
    tgt_model: Type[BertLMHeadModel],
    src_model_list: List[Type[BertLMHeadModel]],
    untie_weights: bool,
    init_type: str,
    skip_layernorm: bool,
):
    global tie_flag, inherit_flag, skip_ln
    
    # overwrite global vars
    tie_flag = not untie_weights
    inherit_flag = init_type in ['pca', 'net2net']
    skip_ln = skip_layernorm
    
    # set init type
    set_init_type(init_type)
    
    # register BertModel
    register_bert(
        tgt_bert=tgt_model.bert,
        src_bert_list=[src_model.bert for src_model in src_model_list],
    )

    # register BertOnlyMLMHead
    # inherit from the last bert layer
    if inherit_flag and not tie_flag:
        a_init_from_prev = [getattr(tgt_model.bert.encoder.layer[-1].output.dense.parametrizations.weight[0], f"e_inv_{i}") for i in range(len(src_model_list))]
    else:
        a_init_from_prev = None
    
    register_mlm_head(
        tgt_mlm_head=tgt_model.cls,
        src_mlm_head_list=[src_model.cls for src_model in src_model_list],
        b_emb=tgt_model.bert.embeddings.word_embeddings.parametrizations.weight[0].ligo_b,
        a_init_from_prev=a_init_from_prev,
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
    

def check_inherit(
    tgt_model:Type[BertLMHeadModel],
    src_model_list: List[Type[BertLMHeadModel]],
    untie_weights: bool,
    init_type: str
):
    global tie_flag

    from pretraining.ligo_parameterization import (
        remove_embedding,
        remove_linear,
        remove_ln,
    )

    set_init_type(init_type)
    n_src_models = len(src_model_list)

    # ===== test embeddings =====
    tgt_word_emb = tgt_model.bert.embeddings.word_embeddings
    src_word_emb_list = [src_model.bert.embeddings.word_embeddings for src_model in src_model_list]
    register_embedding(
        tgt_emb=tgt_word_emb,
        src_emb_list=src_word_emb_list,
        tie_b=None,
    )
    # save b_emb for tying
    b_emb = tgt_word_emb.parametrizations.weight[0].ligo_b
    b_emb_init = [getattr(tgt_word_emb.parametrizations.weight[0], f"e_inv_{i}") for i in range(n_src_models)]

    # # ===== test ln =====
    # tgt_emb_ln = tgt_model.bert.embeddings.LayerNorm
    # src_emb_ln_list = [src_model.bert.embeddings.LayerNorm for src_model in src_model_list]

    # register_ln(
    #     tgt_ln=tgt_emb_ln,
    #     src_ln_list=src_emb_ln_list,
    #     tie_b=None,
    #     bias=tgt_emb_ln.bias is not None,
    #     init_b=b_emb_init,
    # )
    # # remove_ln(tgt_ln, bias=tgt_ln.bias is not None)

    # # ===== test linear (square) =====
    # tgt_linear = tgt_model.bert.encoder.layer[0].attention.self.query
    # src_linear_list = [src_model.bert.encoder.layer[0].attention.self.query for src_model in src_model_list]

    # register_linear(
    #     tgt_linear=tgt_linear,
    #     src_linear_list=src_linear_list,
    #     tie_a=None,
    #     tie_b=None,
    #     bias=tgt_linear.bias is not None,
    #     init_a=b_emb_init,
    # )
    # # remove_linear(tgt_linear, bias=tgt_linear.bias is not None)

    # # ===== test linear (non-square) =====
    # tgt_linear = tgt_model.bert.encoder.layer[0].intermediate.dense_act
    # src_linear_list = [src_model.bert.encoder.layer[0].intermediate.dense_act for src_model in src_model_list]

    # register_linear(
    #     tgt_linear=tgt_linear,
    #     src_linear_list=src_linear_list,
    #     tie_a=b_emb,  # None
    #     tie_b=None,
    #     bias=tgt_linear.bias is not None,
    # )
    # # remove_linear(tgt_linear, bias=tgt_linear.bias is not None)

    # # NOTE: some rows of e_inv_0 are all zero, probably due to zero singular values
    # # (tgt_linear.parametrizations.weight[0].e_inv_0.sum(1) != 0.0).sum() == 1529

    # ===== test linear decoder =====
    tgt_decoder_linear = tgt_model.cls.predictions.decoder
    src_decoder_linear_list = [src_model.cls.predictions.decoder for src_model in src_model_list]
    register_decoder_linear(
        tgt_linear=tgt_decoder_linear,
        src_linear_list=src_decoder_linear_list,
        tie_a=None,
        bias=tgt_decoder_linear.bias is not None,
        init_a=b_emb_init,
    )
    # remove_linear(tgt_decoder_linear, bias=tgt_decoder_linear.bias is not None)

    # # pca checklist
    # # ligo_b == [PCA, I], [I, PCA]
    # tgt_word_emb.parametrizations.weight[0].ligo_b[0]
    # tgt_word_emb.parametrizations.weight[0].ligo_b[1]

    # # e_env_list is registered
    # tgt_word_emb.parametrizations.weight[0].e_inv_0
    # tgt_word_emb.parametrizations.weight[0].e_inv_1

    # # after remove, block diagonal matrix (except for embeddings)
    # remove_embedding(tgt_word_emb)

    breakpoint()
