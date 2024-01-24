import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from transformers.models.siglip import (
    SiglipVisionModel,
    SiglipImageProcessor,
)
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if "mpt" in model_name.lower():
        model = LlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
        
    if "siglip" in args.model_name:
        image_processor = SiglipImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)
    else:
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]
    if vision_tower.device.type == 'meta':
        if 'siglip' in args.model_name:
            vision_tower = SiglipVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        else:
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    
    if "downsample" in model.config.mm_projector_type or "ds" in model.config.mm_projector_type:
        image_token_len = image_token_len // 4
    
    if "p32" in args.model_name:  # extra leading patches
        image_token_len += 32
    elif "se" in model.config.mm_projector_type:
        image_token_len += 2
        
    # read images first
    image_file_list = args.image_file.split("###")
    image_list = [load_image(image_file) for image_file in image_file_list] 
    image_tensor = torch.stack([image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in image_list], dim=0)
    n_images = image_tensor.shape[0]
    print(image_tensor.shape, "image_tensor.shape")

    image_token_len = 4
    query = args.query
    if mm_use_im_start_end:
        image_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN
    else:
        image_tokens = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        
    if not "<image>" in query:
        assert "###" not in query  # single query
        query = image_tokens + "\n" + query  # add <image>
        query_list = [query]
    else: 
        
        query_list = query.split("###")
        assert len(query_list) % 2 == 1  # the last one is from human
        
        new_query_list = []
        for i, query in enumerate(query_list):
            if "<image>" in query:
                assert i % 2 == 0  # only from human
                assert query.startswith("<image>")
                query = query.replace("<image>", image_tokens)
            new_query_list.append(query)
        query_list = new_query_list 

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    
    for i, query in enumerate(query_list):
        conv.append_message(conv.roles[i % 2], query)
    conv.append_message(conv.roles[1], None)
        
    prompt = conv.get_prompt()
    print("%" * 20)
    print(prompt)
    inputs = tokenizer([prompt])
    
    print(tokenizer(["<image>hello world.".replace("<image>", image_tokens)]).input_ids)
    
    exit()

    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            max_new_tokens=256,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    eval_model(args)
