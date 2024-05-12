# This file is modified from https://github.com/haotian-liu/LLaVA/

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


def get_model_output(model, image_processor, tokenizer, video_path, qs, conv_mode="vicuna_v1"):

    num_video_frames = 8

    if "shortest_edge" in image_processor.size:
        image_size = image_processor.size["shortest_edge"]
    else:
        image_size = image_processor.size["height"]

    try:
        # replace with opencv extractor
        # Set a timeout of 5 seconds
        signal.alarm(30)
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = float(video.duration)
        assert duration >= 0.25
        video_outputs = video.get_clip(start_sec=0, end_sec=duration)["video"]
        assert video_outputs.size(1) > 8
        num_frames = video_outputs.shape[1]
        # step = (num_frames - 1) // 8 + 1
        step = num_frames // 8
        num_frames = num_frames - (num_frames % 8)
        indices = torch.floor(torch.arange(0, num_frames, step)).long()
        video_outputs = video_outputs[:, indices, :, :]
        # Cancel the alarm if the code finishes within the timeout
        signal.alarm(0)
    except TimeoutError:
        print(f'Timeout for video path {video_path}')
        video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)
    except Exception as e:
        print(f'bad data path {video_path}')
        print(f"Error processing {video_path}: {e}")
        video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)

    c, b, h, w = video_outputs.size()
    image_tensor = torch.zeros(b, c, image_size, image_size, dtype=torch.uint8)
    video_frames = video_outputs.permute(1, 0, 2, 3).contiguous()
    video_frames = Resize(size=[image_size, image_size], antialias=True)(video_frames)
    image_tensor[:, :, :, :] = video_frames

    image_tensor = [
        image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0)
        for image in torch.unbind(image_tensor)
    ]
    image_tensor = torch.cat(image_tensor, dim=0)
    qs = '<image>\n' * num_video_frames + qs
    
    # num_video_frames = 4
    # image_tensor = image_tensor[0:num_video_frames, ::]
    # qs = '<image>\n' * num_video_frames + qs

    print(image_tensor.shape)
    print(conv_templates, conv_mode, qs)
    # input()
    
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
    
    video_path = osp.expanduser("~/workspace/vila-captioner-avfm/videos/1.mp4")
    question = "please describe the video in details "

    print(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)
    output = get_model_output(model, image_processor, tokenizer, video_path, question)
    print(question)
    print(output)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/video-13b-clip-vit-large-patch14-336-align-llava_1_5_mm_align-pretrain-sharegpt4v_pretrain-SFT-shot2story_shotonly")
    # parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA-7b")
    # parser.add_argument("--model-path", type=str, default="checkpoints/stage2-siglip-large-patch16-384-align-llava_1_5_mm_align-pretrain-sharegpt4v_pretrain-SFT-sharegpt4v_sft+textocr")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, required=False, default=5120)
    # parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    # parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    # parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    # parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
