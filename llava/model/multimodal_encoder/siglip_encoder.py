import torch
from llava.model.multimodal_encoder.vision_encoder import VisionTower

from transformers import PretrainedConfig
from transformers.models.siglip import (
    SiglipVisionModel,
    SiglipImageProcessor,
)


class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path, torch_dtype=eval(config.model_dtype)
        )
        self.is_loaded = True
