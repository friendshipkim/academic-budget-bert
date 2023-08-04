from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StitchArguments:
    """
    Model stitching arguments
    """

    _argument_group_name = "Stitch Arguments"

    do_stitch: Optional[bool] = field(
        default=False, metadata={"help": "whether to stitch two source models"}
    )
    src_model1_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the first source pretrained model (should contain pytorch_model.bin)"
        },
    )
    src_model2_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the second source pretrained model (should contain pytorch_model.bin)"
        },
    )
    src_model3_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Necessary only when stitching 4 models, Path to the third source pretrained model (should contain pytorch_model.bin)"
        },
    )
    src_model4_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Necessary only when stitching 4 models, Path to the second source pretrained model (should contain pytorch_model.bin)"
        },
    )
    finetuned_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the parameterized stitched model (should contain pytorch_model.bin)"
        },
    )
    skip_layernorm: Optional[bool] = field(
        default=False, metadata={"help": "whether to skip stitching layernorms"}
    )
    modularize: Optional[bool] = field(
        default=False, metadata={"help": "whether to use modular linear layers (2d x 2d -> 2 x d x d)"}
    )
    add_blend_layer: Optional[bool] = field(
        default=False, metadata={"help": "for modularized model, whther to add a blend layer at the bottom of BertLayer"}
    )
    add_blend_ln: Optional[bool] = field(
        default=False, metadata={"help": "for modularized model, whther to add a layernorm after the blend layer"}
    )
    overlap: Optional[int] = field(
        default=8, metadata={"help": "decrease the intermediate hidden size from 8x to the given integer too add some blend parameters"}
    )
    num_src_models: Optional[int] = field(
        default=-1, metadata={"help": "(only for ligo) number of source model to stitch"}
    )
    
    # custom target model config
    target_hidden_size: Optional[int] = field(
        default=-1, metadata={"help": "hidden size of the target model"}
    )
    target_num_attention_heads: Optional[int] = field(
        default=-1, metadata={"help": "the number of attention heads of the target model"}
    )
    
    untie_weights: Optional[bool] = field(
        default=False, metadata={"help": "(only for ligo) whether to untie growth operators"}
    )
    avg_logits: Optional[bool] = field(
        default=False, metadata={"help": "whether to average logit values during finetuning and after stitching"}
    )
    init_type: Optional[str] = field(
        default="eye", metadata={"help": "(only for ligo) how to initialize growth operators"}
    )
    profile_model: Optional[bool] = field(
        default=False, metadata={"help": "whether to profile a model, evaluate on 3 batches [10, 20, 30] by default"}
    )
    record_gradient_norm: Optional[bool] = field(
        default=False, metadata={"help": "for stitched model, whether to record l2 norm of gradients of pretrained/epsilon"}
    )
