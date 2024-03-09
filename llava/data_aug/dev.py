import os, os.path as osp, sys
from tqdm import tqdm
import json
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from filelock import Timeout, FileLock
import shutil


# model_id = "NousResearch/Llama-2-13b-chat-hf"
# model_id = "NousResearch/Llama-2-7b-hf"
# Mixtral-8x7B-Instruct-v0.1
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
local_rank = 2

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "load_in_4bit": False,
        "device_map": f"cuda:{local_rank}",
    },  # "device_map": "auto"},
    return_full_text=False,
    repetition_penalty=1.0,
)

generation_config = {
    "temperature": 0.2,
    "top_p": 0.6,
    "do_sample": True,
    "max_new_tokens": 256,
}

print(model_id)
print(generation_config)

while True:
    print("--" * 50)
    # input_msg = input("Please enter inputs:\n")
    input_msg = '''Please reverse the order of words in the sentence. 
For example,
“the more you buy, the more you save” will become “save you more the, buy you more the”
“I love the Micro conference” will become “conference Micro the love I”
Next, please reverse the sentence: “I love Boston and MIT”
    '''
    result = pipe(input_msg + "\n", **generation_config)
    print("--" * 50)
    print(result[0]["generated_text"])
    break