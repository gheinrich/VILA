# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
from parameterized import parameterized
from transformers import AutoTokenizer

from llava.model import LlavaLlamaConfig, LlavaLlamaModel
from llava.train.args import DataArguments, ModelArguments, TrainingArguments
from llava.train.utils import prepare_config_for_training

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    device = "cuda:0"
else:
    device = "cpu"

MODELS = ["lmsys/vicuna-7b-v1.5", "Qwen/Qwen2-7B"]


class TestInputPacking(unittest.TestCase):
    @parameterized.expand(MODELS)
    def test_loss_close(self, model_name_or_path: str):
        if torch.cuda.is_available():
            rank = 0
            torch.cuda.set_device(rank)

        self.model_args = ModelArguments(
            model_name_or_path=model_name_or_path,
            version="v1",
            vision_tower="openai/clip-vit-large-patch14-336",
            mm_projector="mlp2x_gelu",
            mm_vision_select_layer=-2,
            mm_use_im_patch_token=False,
        )
        self.data_args = DataArguments()
        self.training_args = TrainingArguments(output_dir="", bf16=True)
        self.config = LlavaLlamaConfig.from_pretrained(model_name_or_path)
        prepare_config_for_training(self.config, self.model_args, self.training_args, self.data_args)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=4096,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )

        model = LlavaLlamaModel(config=self.config, attn_implementation="flash_attention_2")
        model.vision_tower = model.vision_tower.to(device)
        model.mm_projector = model.mm_projector.to(device)
        model = model.to(device)
        model.llm.pad_token_id = self.tokenizer.pad_token_id

        data = torch.load("tests/sample_data/test_packing.pth")
        data["input_ids"] = data["input_ids"].to(device)
        data["images"] = data["images"].to(torch.bfloat16).to(device)
        data["attention_mask"] = data["attention_mask"].to(device)
        data["labels"] = data["labels"].to(device)

        model.train()

        output = model(**data, packing=True, use_cache=False)
        output_loss = output.loss.item()
        del output

        target = model(**data, packing=False, use_cache=False)
        target_loss = target.loss.item()
        del target

        self.assertAlmostEqual(output_loss, target_loss, places=2)


if __name__ == "__main__":
    unittest.main()
