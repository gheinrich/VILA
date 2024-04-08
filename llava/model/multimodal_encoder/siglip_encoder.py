import torch
from llava.model.multimodal_encoder.vision_encoder import VisionTower

from transformers import AutoConfig, PretrainedConfig, AutoModel
from .siglip import (
    SiglipVisionConfig, 
    SiglipVisionModel,
    SiglipImageProcessor,
)


class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, state_dict=None):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            # TODO(ligeng): why pass config here leading to errors?
            model_name_or_path, torch_dtype=eval(config.model_dtype), state_dict=state_dict
        )
        self.is_loaded = True

AutoConfig.register("siglip_vision_model", SiglipVisionConfig)
AutoModel.register(SiglipVisionConfig, SiglipVisionModel)