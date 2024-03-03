from llava.model.multimodal_encoder.vision_encoder import VisionTower
from llava.model.utils import maybe_resize_pos_embeds

from transformers import PretrainedConfig
from transformers.models.siglip import (
    SiglipVisionModel,
    SiglipImageProcessor,
    SiglipVisionConfig,
)


class SiglipVisionTower(VisionTower):
    def __init__(self, vision_tower_cfg: str | dict, config: PretrainedConfig):
        super().__init__(vision_tower_cfg, config)
        ## build from model_name_or_path
        if isinstance(vision_tower_cfg, str):
            self.image_processor = SiglipImageProcessor.from_pretrained(
                vision_tower_cfg
            )
            self.vision_tower = SiglipVisionModel.from_pretrained(vision_tower_cfg)
            ## resize position embedding of the vision encoder
            maybe_resize_pos_embeds(
                self.image_processor,
                self.vision_tower,
                config.vision_resolution,
                config.interpolate_mode,
            )
        ## build from saved checkpoint
        elif isinstance(vision_tower_cfg, dict):
            assert (
                getattr(config, "resume_path", None) is not None
            ), "You are loading from a checkpoint, but resume_path is None!"
            self.image_processor = SiglipImageProcessor.from_pretrained(
                config.resume_path
            )
            vision_tower_cfg = SiglipVisionConfig.from_dict(vision_tower_cfg)
            self.vision_tower = SiglipVisionModel(vision_tower_cfg)
        self.is_loaded = True
