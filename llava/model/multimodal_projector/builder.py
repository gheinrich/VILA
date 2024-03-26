# This file is modified from https://github.com/haotian-liu/LLaVA/

import torch.nn as nn
import os

from base_projector import MultimodalProjectorConfig, MultimodalProjector
from transformers import PretrainedConfig, PreTrainedModel


def build_mm_projector(
    model_type_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    if model_type_or_path is None:
        return None
    ## load from pretrained model
    if os.path.exists(model_type_or_path):
        return MultimodalProjector.from_pretrained(model_type_or_path)
    ## build from config
    else:
        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        return MultimodalProjector(mm_projector_cfg, config)
