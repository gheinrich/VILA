import math
import warnings
import os, os.path as osp
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
)


def has_tokenizer(path):
    if (
        osp.exists(osp.join(path, "special_tokens_map.json"))
        and osp.exists(osp.join(path, "tokenizer_config.json"))
        and osp.exists(osp.join(path, "tokenizer.model"))
    ):
        # print("[has_tokenizer]", path, True)
        return True
    from huggingface_hub import HfApi, file_exists
    from huggingface_hub.utils import validate_repo_id, HFValidationError
    api = HfApi()
    try:
        valid_hf_repo = api.repo_exists(path)
    except HFValidationError as e:
        valid_hf_repo = False
    # print("DEBUG1", f"[{path}]", valid_hf_repo); input()
    if (
        valid_hf_repo
        and file_exists(path, "special_tokens_map.json")
        and file_exists(path, "tokenizer_config.json")
        and file_exists(path, "tokenizer.model")
    ):
        # print("[has_tokenizer]", path, True)
        return True
    # print("[has_tokenizer]", path, False)
    return False


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
    config_cls: PretrainedConfig = None,
    llm_cls: PreTrainedModel = None,
    attn_implementation=None,
    model_max_length=None,
    *args,
    **kwargs,
) -> PreTrainedModel:
    if config_cls is None:
        config_cls = AutoConfig
    if llm_cls is None:
        llm_cls = AutoModelForCausalLM
    ## extra configuration for llm
    llm_cfg = config_cls.from_pretrained(model_name_or_path)
    llm_cfg._attn_implementation = attn_implementation
    llm_cfg.model_max_length = model_max_length
    if model_max_length is not None:
        context_length_extension(llm_cfg)

    # model_dtype = getattr(config, "model_dtype", "torch.float16")
    # if not hasattr(config, "model_dtype"):
    #     warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
    
    llm = llm_cls.from_pretrained(
        model_name_or_path, config=llm_cfg, torch_dtype=eval(config.model_dtype), *args, **kwargs
    )
    # TODO(ligeng): is this necessary for llava?
    config.hidden_size = llm.config.hidden_size
    
    vlm_cfg = config.resume_path if config.resume_path else config._name_or_path
    
    if has_tokenizer(vlm_cfg):
        warnings.warn("tokenizer found in VLM root folder. Move to MODEL_PATH/llm in the future.")
        tokenizer = AutoTokenizer.from_pretrained(vlm_cfg)
    elif has_tokenizer(llm_cfg):
        tokenizer = AutoTokenizer.from_pretrained(llm_cfg)
    else:
        raise FileNotFoundError(f"Tokenizer not found in the model path.  {vlm_cfg} and {llm_cfg}")
        
    return llm, tokenizer