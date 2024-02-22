from llava.model.multimodal_encoder.vision_encoder import VisionTower
from transformers.models.siglip import (
    SiglipVisionConfig,
    SiglipVisionModel,
    SiglipImageProcessor,
)


class SiglipVisionTower(VisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True