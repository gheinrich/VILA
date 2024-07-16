# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import hashlib
import json
import math
import os
import signal

import torch
from datasets import load_dataset
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


# This function will be called when the timeout is reached
def handler(signum, frame):
    raise TimeoutError()


# Set the signal handler
signal.signal(signal.SIGALRM, handler)


def split_dataset(dataset, num_chunks):
    """Split a dataset into n (roughly) equal-sized chunks"""
    num_samples = len(dataset)
    chunk_size = math.ceil(num_samples / num_chunks)
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk = dataset.select(range(start_idx, end_idx))
        chunks.append(chunk)
    return chunks


def get_chunk(dset, n, k):
    chunks = split_dataset(dset, n)
    return chunks[k]


def dict_to_hash(d):
    """Convert a sample to a hash string to generate the id"""

    def convert(value):
        if isinstance(value, list):
            return tuple(convert(v) for v in value)
        elif isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    converted_dict = convert(d)
    dict_str = json.dumps(converted_dict, sort_keys=True)
    return hashlib.sha256(dict_str.encode("utf-8")).hexdigest()


def format_question_and_options(question, options):
    """
    Formats a question and a list of options into a single string with options labeled A, B, C, etc.

    Parameters:
    - question (str): The question to be formatted.
    - options (list of str): The options for the question.

    Returns:
    - str: The formatted question and options.
    """
    formatted_string = f"{question}\n"
    option_labels = [chr(ord("A") + i) for i in range(len(options))]  # Generate option labels dynamically

    for label, option in zip(option_labels, options):
        formatted_string += f"- {label}) {option}\n"

    return formatted_string


vision_and_language_dependence_prompt = """You will be provided with subtitles from a specific scene of a movie and a few frames from that scene. After going through the movie scene and seeing the frames, please answer the question that follows. The question will have five possible answers labeled A, B, C, D, and E, please try to provide the most probable answer in your opinion. Your output should be just one of A,B,C,D,E and nothing else.

**Subtitles:** \n{subs}\n\nQuestion: {question}"""


def get_prompt(data):
    formatted_subs = data["subtitles"]
    options = data["choices"]
    formatted_question = format_question_and_options(data["question"], options)

    prompt = vision_and_language_dependence_prompt.format(subs=formatted_subs, question=formatted_question)
    return prompt


@torch.inference_mode()
def get_model_output(args, question, images, num_video_frames, model, image_processor, tokenizer):
    images = process_images(images, image_processor, model.config)
    num_frames_loaded_successfully = len(images)
    # print(f"Number of frames loaded successfully: {num_frames_loaded_successfully}")

    prompts = [f"{DEFAULT_IMAGE_TOKEN}\n" for _ in range(num_frames_loaded_successfully)]
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
    # print(output_text)
    output_text = output_text.strip()
    if output_text.endswith(stop_str):
        output_text = output_text[: -len(stop_str)]
    output_text = output_text.strip()

    if output_text in args.options:
        pred_text = output_text
    else:
        pred_text = output_text[0]

    return output_text, pred_text


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)
    args.image_processor = image_processor

    cinepile = load_dataset("tomg-group-umd/cinepile")
    data = get_chunk(cinepile["test"], args.num_chunks, args.chunk_idx)
    video_dir = "/home/xiuli/workspace/cinepile/yt_videos"
    category_mappings = {
        "Character and\nRelationship Dynamics": "CRD",
        "Narrative and\nPlot Analysis": "NPA",
        "Setting and\nTechnical Analysis": "STA",
        "Temporal": "TEMP",
        "Theme Exploration": "TH",
    }

    args.output_dir = os.path.expanduser(args.output_dir)
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    # Read cache answer file, each line is a json object
    if os.path.exists(answers_file):
        cache_ans_file = open(answers_file)
        cache_ans = cache_ans_file.readlines()
        cache_ans_file.close()
    else:
        cache_ans = []

    # Get cached video ids
    cache_set = {json.loads(line)["id"] for line in cache_ans}

    num_video_frames = model.config.num_video_frames
    # print(f"num_video_frames {num_video_frames}")
    if hasattr(model.config, "fps") and model.config.fps is not None:
        fps = model.config.fps
    else:
        fps = 0.0

    match_cnt, total_cnt = 0, 0
    for i, sample in enumerate(tqdm(data)):
        id = dict_to_hash(sample)
        if id in cache_set:
            print(f"Skipping {id} because it is in the cache")
            continue
        video_name = sample["yt_clip_link"].split("watch?v=")[-1] + ".mp4"
        if not os.path.exists(os.path.join(video_dir, video_name)):
            print(f"Video {video_name} does not exist")
            continue

        question = get_prompt(sample) + "\n"
        sample_set = {"id": id, "question": format_question_and_options(sample["question"], sample["choices"])}

        images, video_loading_succeed = LazySupervisedDataset._load_video(
            os.path.join(video_dir, video_name), num_video_frames, fps, args
        )
        if video_loading_succeed == 0:
            print(f"Failed to load video {video_name}")
            continue

        output_text, pred_text = get_model_output(
            args, question, images, num_video_frames, model, image_processor, tokenizer
        )

        ans_key_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        answer = ans_key_map[sample["answer_key_position"]]
        match_cnt += pred_text == answer
        total_cnt += 1

        sample_set["answer"] = answer
        sample_set["pred"] = output_text
        sample_set["question_category"] = category_mappings[sample["question_category"]]
        with open(answers_file, "a") as f:
            f.write(json.dumps(sample_set) + "\n")
        # print(f"pred answer: {pred_text}, gt answer: {answer}")
        # print(f"Total: {total_cnt}, Correct: {match_cnt}, Accuracy: {match_cnt/total_cnt*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", help="Directory to save the model results file.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results file.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--options", type=list, default=["A", "B", "C", "D", "E"])
    args = parser.parse_args()

    eval_model(args)
