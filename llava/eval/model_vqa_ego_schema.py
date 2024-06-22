# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import json
import os
import math

import torch
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.data import LazySupervisedDataset
from llava.eval.mmmu_utils.data_utils import save_json
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    is_gemma_tokenizer,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

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


@torch.inference_mode()
def get_model_output(args, question, images, num_video_frames, model, image_processor, tokenizer):
    images = process_images(images, image_processor, model.config)
    prompts = [f"{DEFAULT_IMAGE_TOKEN}\n" for _ in range(num_video_frames)]
    qs = "".join(prompts) + question

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = (
        [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
        if conv.version == "v0" or is_gemma_tokenizer(tokenizer)
        else None
    )

    output_ids = model.generate(
        input_ids,
        images=images.half().cuda(),
        # images=images.bfloat16().cuda(),
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        max_new_tokens=1024,
        use_cache=True,
        stopping_criteria=stopping_criteria,
    )
    
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(output_text)
    output_text = output_text.strip()
    if output_text.endswith(stop_str):
        output_text = output_text[: -len(stop_str)]
    output_text = output_text.strip()
    
    if output_text in args.options:
        pred_text = output_text
    elif len(output_text) >= 2 and output_text[0] in args.options and output_text[1] == ".":
        pred_text = output_text[0]
    else:
        pred_text = ""
    
    return output_text, pred_text


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)
    args.image_processor = image_processor

    questions = json.load(open(os.path.join(args.question_file), "r"))
    if args.split == "validation":
        answers = json.load(open(os.path.join(args.gt_answers_file), "r"))
        answers = list((key, value) for key, value in answers.items())
    else:
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    video_dir = args.video_folder

    questions = {question["q_uid"]: question for question in questions}
    
    args.output_dir = os.path.expanduser(args.output_dir)
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    # Read cache answer file, each line is a json object
    if os.path.exists(answers_file):
        cache_ans_file = open(answers_file, "r")
        cache_ans = cache_ans_file.readlines()
        cache_ans_file.close()
    else:
        cache_ans = []
        
    # Get cached video ids
    cache_set = set([json.loads(line)['id'] for line in cache_ans])


    num_video_frames = model.config.num_video_frames
    print(f"num_video_frames {num_video_frames}")
    if hasattr(model.config, "fps") and model.config.fps is not None:
        fps = model.config.fps
    else:
        fps = 0.0

    if args.split == "validation":
        match_cnt, total_cnt = 0, 0
        for i, (q_uid, answer) in enumerate(tqdm(answers)):
            if f"{q_uid}" in cache_set:
                print(f"Skipping {q_uid} because it is in the cache")
                continue
            video_name = f"{q_uid}.mp4"
            if not os.path.exists(os.path.join(video_dir, video_name)):
                print(f"Video {video_name} does not exist")
                continue
            
            sample = questions[q_uid]
            question = sample["question"] + "\n"
            for i in range(5):
                question = question + chr(ord("A") + i) + ". " + sample[f"option {i}"] + "\n"
            sample_set = {'id': q_uid, 'question': question}
            question = question + "Answer with the option's letter from the given choices directly."
            
            images, video_loading_succeed = LazySupervisedDataset._load_video(
                os.path.join(video_dir, video_name), num_video_frames, fps, args
            )
            if video_loading_succeed == 0:
                print(f"Failed to load video {video_name}")
                continue

            output_text, pred_text = get_model_output(
                args, question, images, num_video_frames, model, image_processor, tokenizer)

            answer = chr(ord("A") + answer)
            match_cnt += pred_text == answer
            total_cnt += 1
            sample_set['answer'] = answer
            sample_set['pred'] = pred_text
            with open(answers_file, 'a') as f:
                f.write(json.dumps(sample_set) + "\n")

        print(f"Total: {total_cnt}, Correct: {match_cnt}, Accuracy: {match_cnt/total_cnt*100:.2f}%")
    else:
        for i, q_uid in enumerate(tqdm(questions)):
            if f"{q_uid}" in cache_set:
                print(f"Skipping {q_uid} because it is in the cache")
                continue
            video_name = f"{q_uid}.mp4"
            if not os.path.exists(os.path.join(video_dir, video_name)):
                print(f"Video {video_name} does not exist")
                continue
            
            sample = questions[q_uid]
            question = sample["question"] + "\n"
            for i in range(5):
                question = question + chr(ord("A") + i) + ". " + sample[f"option {i}"] + "\n"
            sample_set = {'id': q_uid, 'question': question}
            question = question + "Answer with the option's letter from the given choices directly."

            images, video_loading_succeed = LazySupervisedDataset._load_video(
                os.path.join(video_dir, video_name), num_video_frames, fps, args
            )
            if video_loading_succeed == 0:
                print(f"Failed to load video {video_name}")
                continue
            
            output_text, pred_text = get_model_output(
                args, question, images, num_video_frames, model, image_processor, tokenizer)            
            try:
                pred_text = ord(pred_text) - ord("A")
            except:
                print(f"Error in converting {pred_text} to integer")
                pred_text = pred_text
            print(f"output_text: {output_text}, pred_text: {pred_text}")
            sample_set['pred'] = pred_text
            with open(answers_file, 'a') as f:
                f.write(json.dumps(sample_set) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='Directory to save the model results file.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results file.', required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="./playground/data/eval/EgoSchema/videos")
    parser.add_argument("--question-file", type=str, default="./playground/data/eval/EgoSchema/questions.json")
    parser.add_argument("--gt-answers-file", type=str, default="./playground/data/eval/EgoSchema/subset_answers.json")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--options", type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    args = parser.parse_args()

    eval_model(args)
