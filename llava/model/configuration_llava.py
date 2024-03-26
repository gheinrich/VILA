from transformers import PretrainedConfig


class LlavaConfig(PretrainedConfig):
    model_type = "llava"

    def __init__(
        self,
        llm_cfg=None,
        vision_tower_cfg=None,
        mm_projector_cfg=None,
        mm_hidden_size=None,
        resume=False
    ):
        super().__init__()
        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.mm_hidden_size = mm_hidden_size
        self.resume = resume
