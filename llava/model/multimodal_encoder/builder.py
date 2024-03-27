# This file is modified from https://github.com/haotian-liu/LLaVA/

import os
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SiglipVisionTower


def build_vision_tower(
    model_name_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    ## skip vision tower instantiation
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    if config.resume:
        assert os.path.exists(
            model_name_or_path
        ), f"Resume vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = (
        vision_tower_arch if vision_tower_arch is not None else model_name_or_path
    )
    if "clip" in vision_tower_name:
        vision_tower = CLIPVisionTower(model_name_or_path, config)
    elif "siglip" in vision_tower_name:
        vision_tower = SiglipVisionTower(model_name_or_path, config)
    elif "radio" in vision_tower_name:
        from .radio.radio_encoder import RADIOEncoder
        from transformers import CLIPVisionConfig

        vision_tower = RADIOEncoder(config)

        vision_tower.config = CLIPVisionConfig(
            **{
                "hidden_size": vision_tower.hidden_size,
                "image_size": vision_tower.image_size,
                "model_type": "radio_vision_model",
                "num_attention_heads": None,
                "num_channels": 3,
                "num_hidden_layers": None,
                "patch_size": vision_tower.patch_size,
            }
        )
        vision_tower.config._name_or_path = model_name_or_path
    else:
        raise ValueError(f"Unknown vision tower: {model_name_or_path}")

    config.mm_hidden_size = vision_tower.config.hidden_size
    return vision_tower
