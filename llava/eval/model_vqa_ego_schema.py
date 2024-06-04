# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re
import ipdb

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, is_gemma_tokenizer, process_images
from llava.eval.mmmu_utils.data_utils import save_json
from llava.data import LazySupervisedDataset

from PIL import Image
import math


def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if DEFAULT_IMAGE_TOKEN not in qs:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        if mm_use_im_start_end:
            qs.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        return qs


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)
    args.image_processor = image_processor
    
    questions = json.load(open(os.path.join(args.question_file), "r"))
    answers = json.load(open(os.path.join(args.gt_answers_file), "r"))
    video_dir = args.video_folder

    questions = {question["q_uid"]:question for question in questions}
    answers = list((key, value) for key, value in answers.items())

    match_cnt, total_cnt = 0, 0
    num_video_frames = model.config.num_video_frames
    # video_key_frame_interval = model.config.video_key_frame_interval
    # print(f"num_video_frames {num_video_frames}, video_key_frame_interval {video_key_frame_interval}")
    print(f"num_video_frames {num_video_frames}")
    
    if hasattr(model.config, 'fps') and model.config.fps is not None:
        fps = model.config.fps
    else:
        fps = 0.0
    
    out_samples = dict()
    for i, (q_uid, answer) in enumerate(tqdm(answers)):
        sample = questions[q_uid]
        video_name = f"{q_uid}.mp4"
        question = sample["question"]+"\n"
        for i in range(5):
            question = question+chr(ord("A")+i)+". "+sample[f"option {i}"]+"\n"
        question = question+"Answer with the option's letter from the given choices directly."
        answer = chr(ord("A")+answer)

        images, video_loading_succeed = LazySupervisedDataset._load_video(
            os.path.join(video_dir, video_name), num_video_frames, fps, args)
        images = process_images(images, image_processor, model.config)
        # ipdb.set_trace()
        # qs = generate_video_prompt(num_video_frames, video_key_frame_interval=video_key_frame_interval)+question
        # qs = deal_with_prompt(question, model.config.mm_use_im_start_end)
        prompts = [f"{DEFAULT_IMAGE_TOKEN}\n" for _ in range(num_video_frames)]
        qs = "".join(prompts) + question
        

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # ipdb.set_trace()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" or is_gemma_tokenizer(tokenizer) else None

        with torch.inference_mode():
            # ipdb.set_trace()
            output_ids = model.generate(
                input_ids,
                images=images.half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )
        
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        out_samples[q_uid] = output_text
        output_text = output_text.strip()
        if output_text.endswith(stop_str):
            output_text = output_text[:-len(stop_str)]
        output_text = output_text.strip()
        print(output_text)

        if output_text in args.options:
            pred_text = output_text
        elif len(output_text) >= 2 and output_text[0] in args.options and output_text[1] == ".":
            pred_text = output_text[0]
        else:
            pred_text = ""
        match_cnt += pred_text==answer
        total_cnt += 1

        print(f'Total: {total_cnt}, Correct: {match_cnt}, Accuracy: {match_cnt/total_cnt*100:.2f}%')
        
    os.makedirs("/".join(args.output_path.split("/")[:-1]), exist_ok=True)
    save_json(args.output_path, out_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="llava1.5_13b_val.json", help="name of saved json")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="./playground/data/eval/EgoSchema/videos")
    parser.add_argument("--question-file", type=str, default="./playground/data/eval/EgoSchema/EgoSchema/questions.json")
    parser.add_argument("--gt-answers-file", type=str, default="./playground/data/eval/EgoSchema/EgoSchema/subset_answers.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    args = parser.parse_args()

    eval_model(args)
