from transformers import PretrainedConfig


class LlavaConfig(PretrainedConfig):
    model_type = "llava"

    def __init__(
        self,
        llm_cfg=None,
        vision_tower_cfg=None,
        mm_projector_cfg=None,
        mm_hidden_size=None,
        image_aspect_ratio=None,
        mm_use_im_start_end=False,
        mm_projector_lr=None,
        mm_use_im_patch_token=True,
        resume=False
    ):
        super().__init__()
        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.mm_hidden_size = mm_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_projector_lr = mm_projector_lr
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.resume = resume
