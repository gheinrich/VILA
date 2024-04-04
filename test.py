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

# model = AutoModelForCausalLM.from_pretrained("/home/yunhaof/workspace/scripts/ckpts/vila/debug/reproduce/scratch_stable_test1/stage3")
# print(model)
resume_path = "checkpoints/stage3"
config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
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