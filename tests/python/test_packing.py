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

import copy
import unittest

import torch
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


class TestInputPacking(unittest.TestCase):
    def setUp(self):
        # This test is supposed to run on a single GPU
        if torch.cuda.is_available():
            rank = 0
            torch.cuda.set_device(rank)
        model_name_or_path = "lmsys/vicuna-7b-v1.5"
        self.model_args = ModelArguments(
            model_name_or_path=model_name_or_path,
            version="v1",
            vision_tower="openai/clip-vit-large-patch14-336",
            mm_projector="mlp2x_gelu",
            mm_vision_select_layer=-2,
            mm_use_im_patch_token=False,
        )
        self.data_args = DataArguments()
        self.training_args = TrainingArguments(output_dir="")
        self.config = LlavaLlamaConfig.from_pretrained(model_name_or_path)
        prepare_config_for_training(self.config, self.model_args, self.training_args, self.data_args)
        print("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=4096,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )
        print("Initializing LlavaLlamaModel...")
        self.model = LlavaLlamaModel(config=self.config)
        self.model.vision_tower = self.model.vision_tower.to(device)
        self.model.mm_projector = self.model.mm_projector.to(device)
        self.model = self.model.to(device)

        print("Initializing data...")
        data = torch.load("tests/sample_data/test_packing.pth")
        # necessary for model forward
        self.model.llm.pad_token_id = self.tokenizer.pad_token_id
        self.data = data

    def test_loss_close(self):
        print("Preprocessing inputs...")
        data = copy.deepcopy(self.data)
        data["input_ids"] = data["input_ids"].to(device)
        data["images"] = data["images"].to(torch.float16).to(device)
        data["attention_mask"] = data["attention_mask"].to(device)
        data["labels"] = data["labels"].to(device)
        data["position_ids"] = None
        data["past_key_values"] = None

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.model.prepare_inputs_labels_for_multimodal(**data)

        print("Packing inputs...")
        (
            _,
            new_position_ids,
            new_attention_mask,
            _,
            new_inputs_embeds,
            new_labels,
            sorted_seqlens_in_batch,
        ) = self.model.repack_multimodal_data(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        )

        print("Running models...")

        with torch.no_grad():
            # forward results with input packing
            outputs = self.model.llm.forward(
                input_ids=None,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=None,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
                seqlens_in_batch=sorted_seqlens_in_batch,
            )
            # forward results without input packing
            outputs_ref = self.model.llm.forward(
                input_ids=None,
                attention_mask=attention_mask.to(device),
                position_ids=None,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            # loss should be very similar (but not the same due to numerical precision issues)
            loss = outputs.loss
            loss_ref = outputs_ref.loss
            print("loss =", loss, "loss_ref =", loss_ref)
            self.assertAlmostEqual(loss.item(), loss_ref.item(), places=2)


if __name__ == "__main__":
    unittest.main()
