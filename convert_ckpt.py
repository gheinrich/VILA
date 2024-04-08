import os.path as osp
from collections import OrderedDict
from glob import glob

from safetensors import safe_open
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM,
                          PretrainedConfig, PreTrainedModel)

import llava.model.language_model.llava_llama
from llava.model import *
from llava.model.configuration_llava import LlavaConfig
from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.multimodal_encoder.builder import (SiglipVisionTower,
                                                    build_vision_tower)
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.utils import get_model_config
from llava.model.multimodal_encoder.siglip import (
    SiglipVisionConfig, 
    SiglipVisionModel,
    SiglipImageProcessor,
)

import json

def main(
    path = "~/workspace/VILA/checkpoints/Llama-2-7b-hf-google/siglip-large-patch16-384-align-llava_1_5_mm_align",
    output_dir = "checkpoints/converted_models"
):
    path = osp.expanduser(path)
    # assuming 7b llama + siglip
    config = AutoConfig.from_pretrained("CI-new-format-llama7b-siglip")
    model = AutoModel.from_config(config)
    
    # kep mapping
    state_dict = {} 
    def fn(k):
        if k.startswith("model.layers") or k.startswith("model.norm") or k.startswith("model.embed_tokens") or k.startswith("lm_head"):
            # llm layer
            new_k = "llm." + k
            return new_k
        if k.startswith("model.vision_tower.vision_tower.vision_model."):
            new_k = k.replace("model.vision_tower.vision_tower.vision_model.", "vision_tower.vision_tower.vision_model.")
            return new_k
        if k.startswith("model.mm_projector"):
            new_k = k.replace("model.mm_projector.", "mm_projector.layers.")
            return new_k
        return k

    for sf in glob(osp.join(path, "*.safetensors")):
        with safe_open(sf, framework="pt") as f:
            for key in f.keys():
                state_dict[fn(key)] = f.get_tensor(key)

    for k in state_dict.keys():
        assert k in model.state_dict().keys()

    
    model.load_state_dict(state_dict)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    import fire
    fire.Fire(main)