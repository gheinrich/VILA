# This file is modified from https://github.com/haotian-liu/LLaVA/

import os
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SiglipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif is_absolute_path_exists or "siglip" in vision_tower:
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif is_absolute_path_exists or "radio" in vision_tower:
        from .radio.radio_encoder import RADIOEncoder
        from transformers import CLIPVisionConfig

        vision_tower_cfg.mm_vision_tower = vision_tower
        vision_tower = RADIOEncoder(vision_tower_cfg)

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
        return vision_tower

    raise ValueError(f'Unknown vision tower: {vision_tower}')