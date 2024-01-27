'''
Inference test to run all examples from the paper and compare w/ expected output.
Both the inference results and expected output will be printed out.

Currently do not support multi-turn chat. Each time an image and question are input and answer is output.

Example command line:
srun --label -A llmservice_nlp_fm -N 1 -p interactive -t 2:00:00 -J llmservice_nlp_fm:test5 --gpus-per-node 8 --exclusive  --pty bash ~/workspace/VILA/inference_test/inference_test.sh ~/workspace/VILA/checkpoints/vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-bsz512/
'''


import argparse
import os
import json
import torch

from llava.conversation import SeparatorStyle, conv_templates
from llava.eval.utils import preprocess_image
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"


from llava.eval import run_llava


def eval_model(args, model, tokenizer, image_processor, image_token_len):
    # read json file
    with open(args.test_json_path) as f:
        all_test_cases = json.load(f)
    for test_case in all_test_cases['test_cases']:
        # read images first
        image_file_list = test_case["image_paths"]
        image_list = [run_llava.load_image(os.path.join(args.test_image_path, image_file)) for image_file in image_file_list]
        image_tensor = torch.cat(
            [
                preprocess_image(img, image_processor, use_padding=args.pad)
                for img in image_list
            ],
            dim=0,
        )

        image_tokens = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

        for i in range(len(test_case["QAs"])):
            query = test_case["QAs"][i]["question"]
            query = query.replace("<image>", image_tokens)

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            
            print("%"*10+" "*5+"VILA Response"+" "*5+"%"*10)

            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            outputs = run_llava.process_outputs(args, model, tokenizer, input_ids, image_tensor, stopping_criteria, stop_str)
            print(f'VILA output: {outputs}')
            print(f'Expected output: {test_case["QAs"][i]["expected_answer"]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--test_json_path", type=str, default=None)
    parser.add_argument("--test_image_path", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--pad", action="store_true")

    args = parser.parse_args()

    model, tokenizer, image_processor, image_token_len = run_llava.load_model(args)
    eval_model(args, model, tokenizer, image_processor, image_token_len)

