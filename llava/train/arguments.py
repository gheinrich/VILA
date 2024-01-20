from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    trust_remote_code: bool = field(default=True)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_vision_encoder: bool = field(default=False)
    tune_layer_norm: bool = field(default=False)
    tune_self_attn: bool = field(default=False)
    tune_ffn: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    mm_projector_type: Optional[str] = field(default="linear")
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    textual_embed_path: Optional[str] = field(default=None)
    min_max_range_path: Optional[str] = field(default=None)
    # lora-related
    use_lora: bool = field(default=False)
    # patch attention score
    add_visual_attn_scale: bool = False
    add_visual_expert_mlp: bool = False
    add_visual_expert_attn: bool = False
    predict_image_token: bool = False
    # neftune
    neftune_alpha: float = 0.0


@dataclass
class DataArguments:
    datasets_mixture_name: str
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sep_image_conv_front: bool = False
    image_token_len: int = 0
    image_aspect_ratio: str = "square"
    num_shots: int = 0  # whether to add in-context samples for training


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_self_attn: bool = field(default=False)
    freeze_first_half: bool = field(default=False)
    freeze_later_half: bool = field(default=False)
    train_second_half: bool = field(default=False)
    projector_lr10: bool = field(default=False)
    train_visual_expert_only: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    total_time_limit: int = field(default=-1, metadata={"help": "Timeout limit for this job (in minutes)."})
    pre_terminate_time: int = field(default=10, metadata={"help": "Time to terminate the task inadvance (minutes), saveing checkpoints needs time."})