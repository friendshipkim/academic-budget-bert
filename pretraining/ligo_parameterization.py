import torch
import copy
import math
import torch.nn.utils.parametrize as parametrize

from typing import List, Type
from torch import nn
from torch.nn import Parameter, ParameterList
from pretraining.pca_utils import expand_out_pca_diag, expand_out_pca_separately


# ====================================
# Ligo init functions
# ====================================


def init_eye(params: Type[ParameterList]):
    if len(params) == 1:
        nn.init.eye_(params[0])
    elif len(params) == 2:
        # p: new_dim x dim
        new_dim, dim = params[0].shape
        
        # 1. overlap
        disjoint_span = new_dim - dim
        
        # first ligo: [I, 0] or [I, 1/2, 0]
        param_0 = torch.eye(new_dim, dim)
        param_0[disjoint_span:dim, :] = param_0[disjoint_span:dim, :] / 2
        params[0] = Parameter(param_0, requires_grad=True)

        # second ligo: [0, I] or [0, 1/2, I]
        param_1 = torch.concat([torch.zeros(new_dim - dim, dim), torch.eye(dim)], dim=0)
        param_1[-dim:-disjoint_span, :] = param_1[-dim:-disjoint_span, :] / 2
        params[1] = Parameter(param_1, requires_grad=True)

        # # 2. full A, part of B
        # # first ligo: [I (full), 0]
        # nn.init.eye_(params[0])
        
        # # second ligo: [0, I (part)]
        # keep_last_b_span = new_dim - dim
        # diag_rows = new_dim - torch.arange(keep_last_b_span) - 1
        # diag_cols = dim - torch.arange(keep_last_b_span) - 1
        
        # param_1 = torch.zeros(new_dim, dim)
        # param_1[diag_rows, diag_cols] = torch.ones(keep_last_b_span)
        # params[1] = Parameter(param_1, requires_grad=True)
    
    else:
        raise NotImplementedError


def init_normal(params: Type[ParameterList]):
    for p in params:
        nn.init.normal_(p, mean=0.0, std=0.001)
    
    
def init_kaiming(params: Type[ParameterList]):
    for p in params:
        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        

def init_net2net(params: Type[ParameterList]):
    # TODO: implement net2net initialization
    return

        
def init_net2net_a(params: Type[ParameterList], previous_ligo_b: Type[ParameterList]):
    # TODO: implement stitching
    assert len(params) == 1, f"net2net should have only one source model, but got {len(params)}"
    assert params[0].shape == previous_ligo_b[0].shape
    
    new_in_dim, in_dim = params[0].shape
    in_expand = previous_ligo_b[0].detach()
    
    # frequency of each row copied
    D = 1 / in_expand.sum(-1)  # CHECK: normalization?
    in_expand = torch.diag(D) @ in_expand
    params[0] = Parameter(in_expand, requires_grad=True)


def init_net2net_b(params: Type[ParameterList]):
    # TODO: implement stitching
    assert len(params) == 1, f"net2net should have only one source model, but got {len(params)}"
    new_out_dim, out_dim = params[0].shape
    out_expand = torch.concat((torch.eye(out_dim), torch.eye(out_dim)[torch.randint(out_dim, (new_out_dim - out_dim,))].T), dim=-1)
    params[0] = Parameter(out_expand, requires_grad=True)


def init_pca_a(params: Type[ParameterList], e_inv_list: List[Type[torch.Tensor]]):
    # inherit from the previous layer
    # if previous layer was square, same as ligo_b, different otherwise
    assert len(params) == len(e_inv_list) == 2, f"pca should have two source models, but got {len(params)}"
    
    # init params with e_invs + zeros
    pad_param_list_from_pca_init(params, e_inv_list)


def init_pca_b(params: Type[ParameterList], src_weights: List[Type[torch.Tensor]]):
    assert len(params) == len(src_weights) == 2, f"pca should have two source models, but got {len(params)}"
    
    # dimensions
    # param: [new_out_dim, out_dim]
    # src_weight: [out_dim , in_dim]
    assert params[0].shape[1] == src_weights[0].shape[0]
    new_out_dim, out_dim = params[0].shape
    
    # PCA on each weight metrix
    e_list, e_inv_list = expand_out_pca_separately(src_weights[0], src_weights[1], d_out_new=new_out_dim)
    
    # Transpose e_invs for future use
    e_inv_list = [e_inv.T for e_inv in e_inv_list]
    
    # init params with es + zeros
    pad_param_list_from_pca_init(params, e_list)
    
    return e_inv_list


def pad_param_list_from_pca_init(params: Type[ParameterList], inits: List[Type[torch.Tensor]]):
    assert len(params) == len(inits) == 2, f"pca should have two source models, but got {len(params)}"
    
    # dimensions
    # param: [new_dim, dim]
    # init: [new_dim // 2, dim]
    assert params[0].shape[0] == inits[0].shape[0] * 2
    assert params[0].shape[1] == inits[0].shape[1]
    new_dim, dim = params[0].shape
    
    # first ligo: [PCA, 0]
    nn.init.zeros_(params[0])
    params[0].data[:new_dim // 2, :] = inits[0].data[:]
    # param_0 = torch.concat((inits[0], torch.zeros(new_dim // 2, odimut_dim)), dim=0)
    # params[0] = Parameter(param_0, requires_grad=True)
    
    # second ligo: [0, PCA]
    nn.init.zeros_(params[1])
    params[1].data[-new_dim // 2:, :] = inits[1].data[:]
    # param_1 = torch.concat((torch.zeros(new_dim // 2, dim), inits[1]), dim=0)
    # params[1] = Parameter(param_1, requires_grad=True)

        
init_func_dict = {
    'eye': init_eye,
    'normal': init_normal,
    'kaiming': init_kaiming,
    'net2net_a': init_net2net_a,
    'net2net_b': init_net2net_b,
    'pca_a': init_pca_a,
    'pca_b': init_pca_b,
}

init_type = None


def set_init_type(init_type_):
    global init_type
    init_type = init_type_


# ====================================
# Ligo classes
# ====================================


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
            # init ligo_b depending on init_type
            # if init_type is pca or net2net, e_inv should be returned and registered
            if init_type == 'pca':
                e_inv_list = init_func_dict['pca_b'](
                    self.ligo_b,
                    # NOTE: embedding weights should be transposed for pca
                    src_weights=[src_module_list[i].weight.detach().T for i in range(self.num_src_models)]
                )
                for i, t in enumerate(e_inv_list):
                    self.register_buffer(f"e_inv_{i}", t)
            elif init_type == 'net2net':
                e_inv_list = init_func_dict['net2net_b'](self.ligo_b)
                for i, t in enumerate(e_inv_list):
                    self.register_buffer(f"e_inv_{i}", t)
            else:
                init_func_dict[init_type](self.ligo_b)
        
        # tie ligo_b to given parameter
        else:
            self.ligo_b = tie_b

        # source weights: [vocab_size x d1_out] x n_src
        src_weight = [src_module_list[i].weight.detach() for i in range(self.num_src_models)]
        for i, p in enumerate(src_weight):
            self.register_buffer(f"src_weight_{i}", p)

    @property
    def params(self):
        params = [getattr(self, f"src_weight_{i}") for i in range(self.num_src_models)]
        if init_type in ['pca', 'net2net']:
            params += [getattr(self, f"e_inv_{i}") for i in range(self.num_src_models)]
        return params

    def forward(self, X):
        outputs = []
        for i in range(self.num_src_models):
            outputs.append(torch.mm(getattr(self, f"src_weight_{i}"), self.ligo_b[i].T))

        return torch.stack(outputs, dim=0).sum(0)


class LigoDecoderLinearWeight(nn.Module):
    def __init__(self, in_dim, src_module_list, tie_a=None, init_a=None):
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
            # init ligo_a depending on init_type
            # if init_type is pca or net2net, e_inv should be given
            if init_type == 'pca':
                init_func_dict['pca_a'](self.ligo_a, init_a)
            elif init_type == 'net2net':
                init_func_dict['net2net_a'](self.ligo_a)
            else:
                init_func_dict[init_type](self.ligo_a)
        
        else:
            self.ligo_a = tie_a

        # source weights: [d1_out x d1_in] x n_src
        # average decoder weights to get the same logit value
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
            WA = torch.mm(getattr(self, f"src_weight_{i}"), self.ligo_a[i].T)
            outputs.append(WA)

        return torch.stack(outputs, dim=0).sum(0)


class LigoLinearWeight(nn.Module):
    def __init__(self, in_dim, out_dim, src_module_list, tie_a=None, tie_b=None, init_a=None):
        super().__init__()
        self.num_src_models = len(src_module_list)
        src_in_dim = src_module_list[0].in_features
        src_out_dim = src_module_list[0].out_features

        # register e_inv if tie_b is not given
        self.register_flag = tie_b is None

        # init or tie ligo_b: [d2_out x d1_out]
        if tie_b is None:
            self.ligo_b = nn.ParameterList(
                [
                    copy.deepcopy(Parameter(torch.Tensor(out_dim, src_out_dim), requires_grad=True))
                    for _ in range(self.num_src_models)
                ]
            )
            
            # init ligo_b depending on init_type
            # if init_type is pca or net2net, e_inv should be returned and registered
            if init_type == 'pca':
                e_inv_list = init_func_dict['pca_b'](
                    self.ligo_b,
                    src_weights=[src_module_list[i].weight.detach() for i in range(self.num_src_models)]
                )
                for i, t in enumerate(e_inv_list):
                    self.register_buffer(f"e_inv_{i}", t)
            elif init_type == 'net2net':
                e_inv_list = init_func_dict['net2net_b'](self.ligo_b)
                for i, t in enumerate(e_inv_list):
                    self.register_buffer(f"e_inv_{i}", t)
            else:
                init_func_dict[init_type](self.ligo_b)
        else:
            self.ligo_b = tie_b

        # init or tie ligo_a: [d2_in x d1_in]
        if tie_a is None:
            self.ligo_a = nn.ParameterList(
                [
                    copy.deepcopy(Parameter(torch.Tensor(in_dim, src_in_dim), requires_grad=True))
                    for _ in range(self.num_src_models)
                ]
            )
            
            # init ligo_a depending on init_type
            # if init_type is pca or net2net, e_inv should be given
            if init_type == 'pca':
                init_func_dict['pca_a'](self.ligo_a, init_a)
            elif init_type == 'net2net':
                init_func_dict['net2net_a'](self.ligo_a)
            else:
                init_func_dict[init_type](self.ligo_a)
        else:
            self.ligo_a = tie_a

        # source weights: [d1_out x d1_in] x n_src
        src_weight = [src_module_list[i].weight.detach() for i in range(self.num_src_models)]
        for i, p in enumerate(src_weight):
            self.register_buffer(f"src_weight_{i}", p)

    @property
    def params(self):
        params = [getattr(self, f"src_weight_{i}") for i in range(self.num_src_models)]
        if self.register_flag and (init_type in ['pca', 'net2net']):
            params += [getattr(self, f"e_inv_{i}") for i in range(self.num_src_models)]
        return params

    def forward(self, X):
        outputs = []
        for i in range(self.num_src_models):
            # B W A^T
            BW = torch.mm(self.ligo_b[i], getattr(self, f"src_weight_{i}"))
            BWA = torch.mm(BW, self.ligo_a[i].T)
            outputs.append(BWA)

        return torch.stack(outputs, dim=0).sum(0)


class LigoLinearBias(nn.Module):
    def __init__(self, out_dim, src_module_list, tie_b=None, init_b=None):
        super().__init__()
        self.num_src_models = len(src_module_list)
        src_out_dim = src_module_list[0].out_features

        src_bias = [src_module_list[i].bias.detach() for i in range(self.num_src_models)]
        for i, p in enumerate(src_bias):
            self.register_buffer(f"src_bias_{i}", p)
        
        # tie_b should be given
        assert tie_b is not None, "tie_b should be given at initialization"
        self.ligo_b = tie_b

        # # NOTE: ligo_b is not registered under parametrizations.bias
        # # tie_b should be given at initialization
        # # self.ligo_b = tie_b
        # if tie_b is None:
        #     self.ligo_b = nn.ParameterList(
        #         [
        #             copy.deepcopy(Parameter(torch.Tensor(out_dim, src_out_dim), requires_grad=True))
        #             for _ in range(self.num_src_models)
        #         ]
        #     )
        #     # init ligo_b depending on init_type
        #     if init_type == 'pca':
        #         pad_param_list_from_pca_init(self.ligo_b, init_b)
        #     elif init_type == 'net2net':
        #         # init_b should be given and ligo_b.shape == init_b.shape
        #         for ligo_b, init_b_ in zip(self.ligo_b, init_b):
        #             ligo_b.data[:] = init_b_.data[:]
        #     else:
        #         init_func_dict[init_type](self.ligo_b)
        # else:
        #     self.ligo_b = tie_b

    @property
    def params(self):
        return [getattr(self, f"src_bias_{i}") for i in range(self.num_src_models)]

    def forward(self, X):
        outputs = []
        for i in range(self.num_src_models):
            # matrix vector multiplication
            outputs.append(torch.mv(self.ligo_b[i], getattr(self, f"src_bias_{i}")))
        return torch.stack(outputs, dim=0).sum(0)


class LigoLN(nn.Module):
    def __init__(self, out_dim, src_module_list, tie_b, is_weight, init_b=None):
        super().__init__()
        self.num_src_models = len(src_module_list)
        src_out_dim = src_module_list[0].weight.size(0)

        self.is_weight = is_weight
        
        if is_weight:
            src_weight = [src_module_list[i].weight.detach() for i in range(self.num_src_models)]
            for i, p in enumerate(src_weight):
                self.register_buffer(f"src_weight_{i}", p)
            
            if tie_b is None:
                self.ligo_b = nn.ParameterList(
                    [
                        copy.deepcopy(Parameter(torch.Tensor(out_dim, src_out_dim), requires_grad=True))
                        for _ in range(self.num_src_models)
                    ]
                )
                
                # init ligo_b depending on init_type
                if init_type in ['pca', 'net2net']:
                    # init_b should be given and ligo_b.shape == init_b.shape
                    for ligo_b, init_b_ in zip(self.ligo_b, init_b):
                        ligo_b.data[:] = init_b_.data[:]
                else:
                    init_func_dict[init_type](self.ligo_b)
            else:
                self.ligo_b = tie_b
        
        # if bias, tie_b should be given
        # NOTE: ligo_b is not registered under parametrizations.bias if tied
        else:
            src_bias = [src_module_list[i].bias.detach() for i in range(self.num_src_models)]
            for i, p in enumerate(src_bias):
                self.register_buffer(f"src_bias_{i}", p)
            
            assert tie_b is not None, "tie_b should be given at initialization"
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
    init_a: List[Type[torch.Tensor]] = None,
):
    if (tie_a is None) and (init_type in ['pca', 'net2net']):
        assert init_a is not None, "tie_a or init_a should be given using pca or net2net"
        assert len(init_a) == len(src_linear_list) == 2, f"init_a should have same length as src_module_list, but got {len(init_a)} and {len(src_linear_list)}"
    
    parametrize.register_parametrization(
        tgt_linear,
        "weight",
        LigoLinearWeight(
            in_dim=tgt_linear.in_features,
            out_dim=tgt_linear.out_features,
            src_module_list=src_linear_list,
            tie_a=tie_a,
            tie_b=tie_b,
            init_a=init_a,
        ),
    )
    # NOTE: bias ligo_b is tied to weight ligo_b, can change to init_b
    if bias:
        parametrize.register_parametrization(
            tgt_linear,
            "bias",
            LigoLinearBias(
                out_dim=tgt_linear.out_features,
                src_module_list=src_linear_list,
                tie_b=tgt_linear.parametrizations.weight[0].ligo_b,
                init_b=None,  # not used for now
            ),
        )


def register_decoder_linear(
    tgt_linear: Type[nn.Linear],
    src_linear_list: List[Type[nn.Linear]],
    tie_a: Type[ParameterList],
    init_a: List[Type[torch.Tensor]] = None,
    bias: bool = True,
):
    # decoder only expands input dimension
    # TODO: if decoder has bias (shape: (vocab_size,)), cannot apply ligo_a
    # average two etc
    
    if (tie_a is None) and (init_type in ['pca', 'net2net']):
        assert init_a is not None, "tie_a or init_a should be given using pca or net2net"
        assert len(init_a) == len(src_linear_list) == 2, f"init_a should have same length as src_module_list, but got {len(init_a)} and {len(src_linear_list)}"
    
    parametrize.register_parametrization(
        tgt_linear,
        "weight",
        LigoDecoderLinearWeight(
            in_dim=tgt_linear.in_features,
            src_module_list=src_linear_list,
            tie_a=tie_a,
            init_a=init_a,
        ),
    )
    if bias:
        raise NotImplementedError("Decoder bias is not implemented")


def register_ln(
    tgt_ln: Type[nn.LayerNorm],
    src_ln_list: List[Type[nn.LayerNorm]],
    tie_b: Type[ParameterList],
    bias: bool = True,
    init_b: List[Type[torch.Tensor]] = None,
):
    if (tie_b is None) and (init_type in ['pca', 'net2net']):
        assert init_b is not None, "tie_b or init_b should be given using pca or net2net"
        assert len(init_b) == len(src_ln_list) == 2, f"init_b should have same length as src_module_list, but got {len(init_b)} and {len(src_ln_list)}"
    
    # Layernorm weight, bias is both 1D tensor
    # register weight
    parametrize.register_parametrization(
        tgt_ln,
        "weight",
        LigoLN(
            out_dim=tgt_ln.normalized_shape[0],
            src_module_list=src_ln_list,
            tie_b=tie_b,
            is_weight=True,
            init_b=init_b,
        ),
    )

    # if bias exists, register it
    # bias ligo_b is tied to weight ligo_b, can change to init_b
    if bias:
        parametrize.register_parametrization(
            tgt_ln,
            "bias",
            LigoLN(
                out_dim=tgt_ln.normalized_shape[0],
                src_module_list=src_ln_list,
                tie_b=tgt_ln.parametrizations.weight[0].ligo_b,
                is_weight=False,
                init_b=None,  # not used for now
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
