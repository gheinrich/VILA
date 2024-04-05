from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
)

from llava.model import *
import llava.model.language_model.llava_llama
from collections import OrderedDict
from llava.model.utils import get_model_config
from llava.model.language_model.builder import build_llm
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.configuration_llava import LlavaConfig

# model = AutoModelForCausalLM.from_pretrained("/home/yunhaof/workspace/scripts/ckpts/vila/debug/reproduce/scratch_stable_test1/stage3")
# print(model)
resume_path = "checkpoints/stage3"
config = LlavaLlamaConfig.from_pretrained(resume_path)
config.resume_path = resume_path
model_cls = eval(config.architectures[0])
# config.model_dtype = "torch.bfloat16"
model = model_cls(
    config=config,
)

exit(0)

model_cls = LlavaLlamaModel
config = LlavaLlamaConfig.from_pretrained(
    "NousResearch/Llama-2-7b-hf"
)

model = model_cls(
    config=config,
)
#     attn_implementation="flash_attention_2",
# )