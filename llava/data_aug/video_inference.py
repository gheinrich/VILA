# This file is modified from https://github.com/haotian-liu/LLaVA/

import glob
import argparse
import torch
import os, os.path as osp
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import numpy as np

from torchvision.transforms import Resize
from pytorchvideo.data.encoded_video import EncodedVideo

import signal

# This function will be called when the timeout is reached
def handler(signum, frame):
    raise TimeoutError()
# Set the signal handler
signal.signal(signal.SIGALRM, handler)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_model_output(model, image_processor, tokenizer, video_path, qs, conv_mode="vicuna_v1", num_video_frames = 8):
    from llava.mm_utils import opencv_extract_frames
    imgs = opencv_extract_frames(video_path, num_video_frames)
    image_tensor = [
        # processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in torch.unbind(image_tensor)
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in imgs
    ]
    image_tensor = torch.stack(image_tensor)
        
    qs = '<image>\n' * num_video_frames + qs
    
    conv = conv_templates[conv_mode].copy()
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    input_ids = torch.unsqueeze(input_ids, 0)
    input_ids = torch.as_tensor(input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    print(model_path)
    video_list = list(glob.glob(osp.expanduser(osp.join(args.video_dir, "*.mp4"))))
    assert len(video_list) > 0, f"no video found in {args.video_dir}"
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)
    
    video_path = video_list[0]
    output_json = {}
    question = "This video shows an ego-centric view of a vehicle driving. Please describe the behavior of the ego vehicle."
    
    for video_path in video_list:
        output = get_model_output(model, image_processor, tokenizer, video_path, question)
        print(f"[{video_path}]", question)
        print(output)
        output_json[video_path] = output

        with open(args.output_name, "w") as fp:
            json.dump(output_json, fp, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/Llama-2-7b-hf-siglip-large-patch16-384-align-llava_1_5_mm_align-pretrain-sharegpt4v_pretrain-SFT-sharegpt4v_sft+vflan")
    # parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA-7b")
    # parser.add_argument("--model-path", type=str, default="checkpoints/stage2-siglip-large-patch16-384-align-llava_1_5_mm_align-pretrain-sharegpt4v_pretrain-SFT-sharegpt4v_sft+textocr")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, required=False, default=5120)
    parser.add_argument('--video_dir', help='Directory containing video files.', default="~/workspace/vila-captioner-avfm/videos")
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', default="video_inference_dev.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
