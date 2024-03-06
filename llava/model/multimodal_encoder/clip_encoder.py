# This file is modified from https://github.com/haotian-liu/LLaVA/
from llava.model.multimodal_encoder.vision_encoder import VisionTower
from transformers import (
    PretrainedConfig,
    CLIPVisionModel,
    CLIPImageProcessor,
    CLIPVisionConfig,
)


class CLIPVisionTower(VisionTower):
    def __init__(self, vision_tower_cfg: str | dict, config: PretrainedConfig):
        super().__init__(vision_tower_cfg, config)
        ## build from model_name_or_path
        if isinstance(vision_tower_cfg, str):
            self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower_cfg)
            self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower_cfg)
        ## build from saved checkpoint
        elif isinstance(vision_tower_cfg, dict):
            assert (
                getattr(config, "resume_path", None) is not None
            ), "You are loading from a checkpoint, but resume_path is None!"
            self.image_processor = CLIPImageProcessor.from_pretrained(
                config.resume_path
            )
            vision_tower_cfg = CLIPVisionConfig.from_dict(vision_tower_cfg)
            self.vision_tower = CLIPVisionModel(vision_tower_cfg)
        self.is_loaded = True
