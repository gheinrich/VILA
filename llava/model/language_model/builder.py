import math
import torch
from transformers import PretrainedConfig, PreTrainedModel


def context_length_extension(config):
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    model_max_length = getattr(config, "model_max_length", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        print(f"Scaling RoPE from {orig_ctx_len} to {model_max_length}")
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}


def build_llm(
    model_name_or_path: str,
    config: PretrainedConfig,
    config_cls: PretrainedConfig,
    llm_cls: PreTrainedModel,
    attn_implementation=None,
    model_max_length=None,
    *args,
    **kwargs,
) -> PreTrainedModel:
    ## extra configuration for llm
    llm_cfg = config_cls.from_pretrained(model_name_or_path)
    llm_cfg._attn_implementation = attn_implementation
    llm_cfg.model_max_length = model_max_length
    if model_max_length is not None:
        context_length_extension(llm_cfg)

    llm = llm_cls.from_pretrained(
        model_name_or_path, config=llm_cfg, torch_dtype=config.model_dtype, *args, **kwargs
    )
    config.hidden_size = llm.config.hidden_size
    return llm
