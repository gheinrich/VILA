"""
A inference test on llava_arch.py
This test can be simply run by
python llava_arch_unit_test.py \
            --model_path path_to_model \
            --question_file path_to_question_file \
            --image_folder image_directory \
            --device "cuda:0"
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Efficient-Large-Model/VILA-7b")
    parser.add_argument("--question_file", type=str, default="tests/sample_data/llava_arch_test.json")
    parser.add_argument("--image_folder", type=str, default="tests/sample_data/llava_arch_test_images")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # model initialization
    device = args.device
    torch.set_default_dtype(torch.float16)

    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, model_name="vila", device=device)
    vision_tower = model.get_vision_tower()
    image_size = vision_tower.config.image_size
    patch_size = vision_tower.config.patch_size
    visual_tokens_per_image = (image_size // patch_size) ** 2

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "").strip()
        cur_prompt = qs

        print("Checking Question: %s" % qs.split("\n")[0])
        if "image" in line:
            image_file = line["image"]
            print("Image file: %s" % image_file)
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            images = image_tensor.unsqueeze(0).half().cuda()

            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            cur_prompt = "<image>" + "\n" + cur_prompt
        else:
            images = None

        input_ids = (
            tokenizer_image_token(cur_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        )

        attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.int64)
        position_ids = torch.arange(input_ids.shape[-1], device=device)

        (
            input_ids_after,
            position_ids_after,
            attention_mask_after,
            _,
            inputs_embeds,
            _,
        ) = model.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, None, None, images)

        if images is None:
            assert (
                position_ids_after - position_ids
            ).abs().sum() == 0, "positions_ids should not be changed, without images"
            assert (
                attention_mask_after - attention_mask
            ).abs().sum() == 0, "attention_mask should not be changed, without images"
            assert (input_ids_after - input_ids).abs().sum() == 0, "input_ids should not be changed without images"
            assert inputs_embeds is None, "inputs_embeds should be None without images"
        else:
            assert position_ids_after.shape == (
                input_ids.shape[0],
                input_ids.shape[1] + visual_tokens_per_image - 1,
            ), "positions_ids should not be changed, without images"
            assert attention_mask_after.shape == (
                input_ids.shape[0],
                input_ids.shape[1] + visual_tokens_per_image - 1,
            ), "attention_mask should not be changed, without images"
            assert input_ids_after is None, "input_ids should not be changed without images"
            assert inputs_embeds.shape == (
                input_ids.shape[0],
                input_ids.shape[1] + visual_tokens_per_image - 1,
                4096,
            ), "inputs_embeds should have shape (batch size, num_tokens, hidden_dim)"

        print("Checking passed.")
