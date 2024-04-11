
from llava.model.multimodal_encoder.vision_encoder import VisionTower
from llava.model.multimodal_encoder.intern.configuring_intern_vit import InternVisionConfig
from llava.model.multimodal_encoder.intern.modeling_intern_vit import InternVisionModel
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform

class InternPreprocessor(object):

    def preprocess(self, image, return_tensors):
        transform = build_transform(448)
        image_tensor = transform(image)
        return {'pixel_values': [image_tensor]}


class InternVisionTower(VisionTower):
    def __init__(self, vision_tower, args, drop_path_rate=0.):
        super().__init__(vision_tower, args)
        self._drop_path_rate = drop_path_rate
        
        self.image_processor = InternPreprocessor()
        vision_config = InternVisionConfig.from_pretrained(vision_tower)
        vision_config.drop_path_rate = self._drop_path_rate
        self.vision_tower = InternVisionModel.from_pretrained(
            vision_tower,
            torch_dtype=eval(vision_config.model_dtype),
            config=vision_config)

        self.is_loaded = True
