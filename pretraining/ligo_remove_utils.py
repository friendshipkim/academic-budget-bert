from typing import Type

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
    remove_embedding,
    remove_linear,
    remove_ln,
)


def remove_bert_embeddings(tgt_emb: Type[BertEmbeddings]):
    # word embeddings
    remove_embedding(tgt_emb.word_embeddings)
    remove_embedding(tgt_emb.position_embeddings)
    remove_embedding(tgt_emb.token_type_embeddings)

    if tgt_emb.config.hf_architecture:
        remove_ln(tgt_ln=tgt_emb.LayerNorm, bias=tgt_emb.LayerNorm.bias is not None)
    

def remove_self_attn(tgt_self_attn: Type[BertSelfAttention]):
    # query/key/value
    remove_linear(
        tgt_linear=tgt_self_attn.query, bias=tgt_self_attn.query.bias is not None
    )
    remove_linear(tgt_linear=tgt_self_attn.key, bias=tgt_self_attn.key.bias is not None)
    remove_linear(
        tgt_linear=tgt_self_attn.value, bias=tgt_self_attn.value.bias is not None
    )


def remove_attn(tgt_attn: Type[BertAttention]):
    # Key, query, value projections
    remove_self_attn(tgt_attn.self)

    # Output projection
    remove_linear(
        tgt_linear=tgt_attn.output.dense, bias=tgt_attn.output.dense.bias is not None
    )


def remove_layer(tgt_layer: Type[BertLayer]):
    # Multihead attentions
    remove_attn(tgt_layer.attention)

    # Intermediate ffn
    remove_linear(
        tgt_linear=tgt_layer.intermediate.dense_act,
        bias=tgt_layer.intermediate.dense_act.bias is not None,
    )

    # Output ffn
    remove_linear(tgt_layer.output.dense, bias=tgt_layer.output.dense.bias is not None)

    # copy both PreAttentionLayerNorm, PostAttentionLayerNorm
    remove_ln(
        tgt_ln=tgt_layer.PreAttentionLayerNorm,
        bias=tgt_layer.PreAttentionLayerNorm.bias is not None,
    )
    remove_ln(
        tgt_ln=tgt_layer.PostAttentionLayerNorm,
        bias=tgt_layer.PostAttentionLayerNorm.bias is not None,
    )


def remove_bert(tgt_bert: Type[BertModel]):
    # ==== remove embeddings
    remove_bert_embeddings(tgt_emb=tgt_bert.embeddings)

    # ===== remove encoder layers
    for layer in tgt_bert.encoder.layer:
        remove_layer(layer)

    # when using hf_architecture, no final LayerNorm and Pooler
    if not tgt_bert.config.hf_architecture:
        # ===== remove final LayerNorm
        remove_ln(
            tgt_ln=tgt_bert.encoder.FinalLayerNorm,
            bias=tgt_bert.encoder.FinalLayerNorm.bias is not None,
        )

        # NOTE: pooler is not used during pretraining
        # ===== remove pooler
        remove_linear(
            tgt_linear=tgt_bert.pooler.dense_act,
            bias=tgt_bert.pooler.dense_act.bias is not None,
        )


def remove_mlm_head(tgt_mlm_head: Type[BertOnlyMLMHead]):
    # remove linear in BertPredictionHeadTransform
    remove_linear(
        tgt_linear=tgt_mlm_head.predictions.transform.dense_act,
        bias=tgt_mlm_head.predictions.transform.dense_act.bias is not None,
    )

    # remove LN in BertPredictionHeadTransform
    remove_ln(
        tgt_ln=tgt_mlm_head.predictions.transform.LayerNorm,
        bias=tgt_mlm_head.predictions.transform.LayerNorm.bias is not None,
    )

    # remove decoder
    # TODO: this should be similar to embedding, no bias (but bias exist in hf)
    # (decoder): Linear(in_features=1024, out_features=30528, bias=False)
    # TODO: what is cls.predictions.bias?
    remove_linear(
        tgt_linear=tgt_mlm_head.predictions.decoder,
        bias=tgt_mlm_head.predictions.decoder.bias is not None,
    )


def remove_models(tgt_model: Type[BertLMHeadModel]):
    # remove BertModel
    remove_bert(
        tgt_bert=tgt_model.bert,
    )

    # remove BertOnlyMLMHead
    remove_mlm_head(
        tgt_mlm_head=tgt_model.cls,
    )
    
    # check if no parameterizations are left
    for param_name in dict(tgt_model.named_parameters()).keys():
        assert "parametrizations" not in param_name
