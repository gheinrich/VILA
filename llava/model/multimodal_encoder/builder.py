# This file is modified from https://github.com/haotian-liu/LLaVA/

from transformers import PretrainedConfig
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SiglipVisionTower

## TODO re-design the vision tower registration

## TODO re-design the vision tower registration


def build_vision_tower(config: PretrainedConfig):
    if getattr(config, "vision_tower_config", None) is None:
        vision_tower_cfg = getattr(config, "vision_tower", None)
    else:
        vision_tower_cfg = config.vision_tower_config
    try:
        vision_tower_name = (
            vision_tower_cfg
            if isinstance(vision_tower_cfg, str)
            else vision_tower_cfg["_name_or_path"]
        )
    except:
        vision_tower_name = None

    if vision_tower_name.startswith("openai"):
        return CLIPVisionTower(vision_tower_cfg, config)
    elif "siglip" in vision_tower_name:
        return SiglipVisionTower(vision_tower_cfg, config)
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
        return vision_tower

    raise ValueError(f"Unknown vision tower: {vision_tower_name}")
