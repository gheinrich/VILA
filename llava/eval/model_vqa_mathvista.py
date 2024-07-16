# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import math
import os
from typing import Dict, Tuple

import torch
from datasets import load_dataset
from mmengine import dump
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import Conversation, SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    is_gemma_tokenizer,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from .mathvista_utils.extract_answer import extract_answer


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process_data(
    data: Dict,
    model_config: PretrainedConfig,
    image_processor: BaseImageProcessor,
    tokenizer: PreTrainedTokenizer,
    conv: Conversation,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ## preprocess image
    image = data["decoded_image"].convert("RGB")
    image_tensor = process_images([image], image_processor, model_config)
    ## preprocess text
    qs = data["query"]
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv.copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    return input_ids, image_tensor


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)

    data = load_dataset(os.path.expanduser(args.data_file))[args.split]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    answer_dict = {}

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )
    ## configure conv template
    conv = conv_templates[args.conv_mode]
    keywords = [conv.sep]

    for d in tqdm(data, total=len(data)):
        pid = d["pid"]
        input_ids, image_tensor = process_data(
            d,
            model_config=model.config,
            image_processor=image_processor,
            tokenizer=tokenizer,
            conv=conv,
        )

        input_ids = input_ids.to(device="cuda", non_blocking=True)

        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
            if args.conv_mode == "v0" or is_gemma_tokenizer(tokenizer)
            else None
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        res = extract_answer(outputs, d)
        d["extraction"] = res
        d["decoded_image"] = ""
        answer_dict[pid] = d

    dump(answer_dict, answers_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-file", type=str, default="AI4Math/MathVista")
    parser.add_argument("--split", type=str, default="testmini")
    parser.add_argument("--answers-file", type=str, default="answers.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
