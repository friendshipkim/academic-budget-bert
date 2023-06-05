import torch
import copy
import math
import torch.nn.utils.parametrize as parametrize

from typing import List, Type
from torch import nn
from torch.nn import Parameter, ParameterList


def init_params_id(params: Type[ParameterList]):                               
    # p: d2_dim x d1_dim
    assert len(params) == 2
    # first ligo: [I, 0]
    params[0] = nn.init.eye_(params[0])
    # second ligo: [0, I]
    d2_dim, d1_dim = params[1].shape
    params[1] = Parameter(torch.flip(nn.init.eye_(params[1]).view(2, d1_dim, d1_dim), dims=[0]).view(d2_dim, d1_dim), requires_grad=True)


def init_params_normal(params: Type[ParameterList]):
    for p in params:
        nn.init.normal_(p, mean=0.0, std=0.001)
    
    
def init_params_kaiming(params: Type[ParameterList]):
    for p in params:
        nn.init.kaiming_uniform_(p, a=math.sqrt(5))


class LigoEmbedding(nn.Module):
    def __init__(
        self,
        out_dim: int,
        src_module_list: List[Type[nn.Embedding]],
        tie_b: Type[ParameterList] = None,
    ):
        super().__init__()
        self.num_src_models = len(src_module_list)
        src_out_dim = src_module_list[0].embedding_dim

        # init ligo operators
        # b: [d2_out x d1_out] x n_src
        if tie_b is None:
            self.ligo_b = ParameterList(
                [
                    copy.deepcopy(Parameter(torch.Tensor(out_dim, src_out_dim), requires_grad=True))
                    for _ in range(self.num_src_models)
                ]
            )
            init_params_id(self.ligo_b)
        else:
            self.ligo_b = tie_b

        # source weights: [vocab_size x d1_out] x n_src
        src_weight = [src_module_list[i].weight.detach() for i in range(self.num_src_models)]
        for i, p in enumerate(src_weight):
            self.register_buffer(f"src_weight_{i}", p)

    @property
    def params(self):
        return [getattr(self, f"src_weight_{i}") for i in range(self.num_src_models)]

    def forward(self, X):
        outputs = []
        for i in range(self.num_src_models):
            # matrix multiplication
            outputs.append(torch.mm(self.params[i], self.ligo_b[i].T))

        return torch.stack(outputs, dim=0).sum(0)


class LigoDecoderLinearWeight(nn.Module):
    def __init__(self, in_dim, src_module_list, tie_a=None):
        super().__init__()
        self.num_src_models = len(src_module_list)
        src_in_dim = src_module_list[0].in_features

        # init or tie ligo operators
        # a: d2_in x d1_in
        if tie_a is None:
            self.ligo_a = nn.ParameterList(
                [
                    copy.deepcopy(Parameter(torch.Tensor(in_dim, src_in_dim), requires_grad=True))
                    for _ in range(self.num_src_models)
                ]
            )
            init_params_id(self.ligo_a)
        else:
            self.ligo_a = tie_a

        # source weights: [d1_out x d1_in] x n_src
        # average decoder weights to get the same logit value
        # NOTE: since this is shared with word_embedding, turn it off for now (/ self.num_src_models)
        src_weight = [src_module_list[i].weight.detach() for i in range(self.num_src_models)]
        for i, p in enumerate(src_weight):
            self.register_buffer(f"src_weight_{i}", p)

    @property
    def params(self):
        return [getattr(self, f"src_weight_{i}") for i in range(self.num_src_models)]

    def forward(self, X):
        outputs = []
        for i in range(self.num_src_models):
            # W A^T
            WA = torch.mm(self.params[i], self.ligo_a[i].T)
            outputs.append(WA)

        return torch.stack(outputs, dim=0).sum(0)


class LigoLinearWeight(nn.Module):
    def __init__(self, in_dim, out_dim, src_module_list, tie_a=None, tie_b=None):
        super().__init__()
        self.num_src_models = len(src_module_list)
        src_in_dim = src_module_list[0].in_features
        src_out_dim = src_module_list[0].out_features

        # init or tie ligo operators
        # b: d2_out x d1_out, a: d2_in x d1_in
        if tie_b is None:
            self.ligo_b = nn.ParameterList(
                [
                    copy.deepcopy(Parameter(torch.Tensor(out_dim, src_out_dim), requires_grad=True))
                    for _ in range(self.num_src_models)
                ]
            )
            init_params_id(self.ligo_b)
        else:
            self.ligo_b = tie_b

        if tie_a is None:
            self.ligo_a = nn.ParameterList(
                [
                    copy.deepcopy(Parameter(torch.Tensor(in_dim, src_in_dim), requires_grad=True))
                    for _ in range(self.num_src_models)
                ]
            )
            init_params_id(self.ligo_a)
        else:
            self.ligo_a = tie_a

        # source weights: [d1_out x d1_in] x n_src
        src_weight = [src_module_list[i].weight.detach() for i in range(self.num_src_models)]
        for i, p in enumerate(src_weight):
            self.register_buffer(f"src_weight_{i}", p)

    @property
    def params(self):
        return [getattr(self, f"src_weight_{i}") for i in range(self.num_src_models)]

    def forward(self, X):
        outputs = []
        for i in range(self.num_src_models):
            # B W A^T
            BW = torch.mm(self.ligo_b[i], self.params[i])
            BWA = torch.mm(BW, self.ligo_a[i].T)
            outputs.append(BWA)

        return torch.stack(outputs, dim=0).sum(0)


class LigoLinearBias(nn.Module):
    def __init__(self, out_dim, src_module_list, tie_b=None):
        super().__init__()
        self.num_src_models = len(src_module_list)
        src_out_dim = src_module_list[0].out_features

        src_bias = [src_module_list[i].bias.detach() for i in range(self.num_src_models)]
        for i, p in enumerate(src_bias):
            self.register_buffer(f"src_bias_{i}", p)

        # NOTE: ligo_b is not registered under parametrizations.bias
        # tie_b should be given at initialization
        # self.ligo_b = tie_b
        if tie_b is None:
            self.ligo_b = nn.ParameterList(
                [
                    copy.deepcopy(Parameter(torch.Tensor(out_dim, src_out_dim), requires_grad=True))
                    for _ in range(self.num_src_models)
                ]
            )
            init_params_id(self.ligo_b)
        else:
            self.ligo_b = tie_b

    @property
    def params(self):
        return [getattr(self, f"src_bias_{i}") for i in range(self.num_src_models)]

    def forward(self, X):
        outputs = []
        for i in range(self.num_src_models):
            # matrix vector multiplication
            outputs.append(torch.mv(self.ligo_b[i], self.params[i]))
        return torch.stack(outputs, dim=0).sum(0)


class LigoLN(nn.Module):
    def __init__(self, out_dim, src_module_list, tie_b, is_weight):
        super().__init__()
        self.num_src_models = len(src_module_list)
        src_out_dim = src_module_list[0].weight.size(0)

        self.is_weight = is_weight

        if self.is_weight:
            src_weight = [src_module_list[i].weight.detach() for i in range(self.num_src_models)]
            for i, p in enumerate(src_weight):
                self.register_buffer(f"src_weight_{i}", p)
        else:
            src_bias = [src_module_list[i].bias.detach() for i in range(self.num_src_models)]
            for i, p in enumerate(src_bias):
                self.register_buffer(f"src_bias_{i}", p)

        # NOTE: ligo_b is not registered under parametrizations.bias
        # tie_b should be given at initialization
        # self.ligo_b = tie_b
        if tie_b is None:
            self.ligo_b = nn.ParameterList(
                [
                    copy.deepcopy(Parameter(torch.Tensor(out_dim, src_out_dim), requires_grad=True))
                    for _ in range(self.num_src_models)
                ]
            )
            init_params_id(self.ligo_b)
        else:
            self.ligo_b = tie_b

    @property
    def params(self):
        if self.is_weight:
            return [getattr(self, f"src_weight_{i}") for i in range(self.num_src_models)]
        else:
            return [getattr(self, f"src_bias_{i}") for i in range(self.num_src_models)]

    def forward(self, X):
        outputs = []
        for i in range(self.num_src_models):
            # matrix vector multiplication
            outputs.append(torch.mv(self.ligo_b[i], self.params[i]))
        return torch.stack(outputs, dim=0).sum(0)


# ====================================
# register functions for Embedding, Linear, and LN
# ====================================


def register_embedding(
    tgt_emb: Type[nn.Embedding],
    src_emb_list: List[Type[nn.Embedding]],
    tie_b: Type[ParameterList] = None,
):
    parametrize.register_parametrization(
        tgt_emb,
        "weight",
        LigoEmbedding(
            out_dim=tgt_emb.embedding_dim,
            src_module_list=[src_emb for src_emb in src_emb_list],
            tie_b=tie_b,
        ),
    )


def register_linear(
    tgt_linear: Type[nn.Linear],
    src_linear_list: List[Type[nn.Linear]],
    tie_a: Type[ParameterList],
    tie_b: Type[ParameterList],
    bias: bool = True,
    is_decoder: bool = False,
):
    # apply ligo to weight
    if not is_decoder:
        parametrize.register_parametrization(
            tgt_linear,
            "weight",
            LigoLinearWeight(
                in_dim=tgt_linear.in_features,
                out_dim=tgt_linear.out_features,
                src_module_list=src_linear_list,
                tie_a=tie_a,
                tie_b=tie_b,
            ),
        )
        if bias:
            parametrize.register_parametrization(
                tgt_linear,
                "bias",
                LigoLinearBias(
                    out_dim=tgt_linear.out_features,
                    src_module_list=src_linear_list,
                    tie_b=tgt_linear.parametrizations.weight[0].ligo_b,
                ),
            )
    else:
        # decoder only expands input dimension
        # TODO: if decoder has bias (shape: (vocab_size,)), cannot apply ligo_a
        # average two etc
        parametrize.register_parametrization(
            tgt_linear,
            "weight",
            LigoDecoderLinearWeight(
                in_dim=tgt_linear.in_features,
                src_module_list=src_linear_list,
                tie_a=tie_a,
            ),
        )
        if bias:
            raise NotImplementedError("Decoder bias is not implemented")


def register_ln(
    tgt_ln: Type[nn.LayerNorm],
    src_ln_list: List[Type[nn.LayerNorm]],
    tie_b: Type[ParameterList],
    bias: bool = True,
):
    # register weight
    parametrize.register_parametrization(
        tgt_ln,
        "weight",
        LigoLN(
            out_dim=tgt_ln.normalized_shape[0],
            src_module_list=src_ln_list,
            tie_b=tie_b,
            is_weight=True,
        ),
    )

    # if bias exists, register it
    if bias:
        parametrize.register_parametrization(
            tgt_ln,
            "bias",
            LigoLN(
                out_dim=tgt_ln.normalized_shape[0],
                src_module_list=src_ln_list,
                tie_b=tgt_ln.parametrizations.weight[0].ligo_b,
                is_weight=False,
            ),
        )


# ====================================
# remove functions for Embedding, Linear, and LN
# ====================================
def remove_embedding(tgt_emb: Type[nn.Embedding]):
    parametrize.remove_parametrizations(tgt_emb, "weight", leave_parametrized=True)


def remove_linear(
    tgt_linear: Type[nn.Linear],
    bias: bool = True,
):
    parametrize.remove_parametrizations(tgt_linear, "weight", leave_parametrized=True)
    if bias:
        parametrize.remove_parametrizations(tgt_linear, "bias", leave_parametrized=True)

        
def remove_ln(
    tgt_ln: Type[nn.Linear],
    bias: bool = True,
):
    parametrize.remove_parametrizations(tgt_ln, "weight", leave_parametrized=True)
    if bias:
        parametrize.remove_parametrizations(tgt_ln, "bias", leave_parametrized=True)
