# This file is modified from https://github.com/haotian-liu/LLaVA/
from llava.model.multimodal_encoder.vision_encoder import VisionTower
from transformers import (
    PretrainedConfig,
    CLIPVisionModel,
    CLIPImageProcessor,
)


class CLIPVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(model_name_or_path)
        self.is_loaded = True
