# This file is modified from https://github.com/haotian-liu/LLaVA/

from transformers import PretrainedConfig, PreTrainedModel
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SiglipVisionTower


def build_vision_tower(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if "clip" in model_name_or_path:
        vision_tower = CLIPVisionTower(model_name_or_path, config)
    elif "siglip" in model_name_or_path:
        vision_tower = SiglipVisionTower(model_name_or_path, config)
    elif "radio" in model_name_or_path:
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
