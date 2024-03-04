# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import requests
import tensorrt as trt
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    NougatProcessor,
    NougatTokenizerFast,
    CLIPVisionModel,
)

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.runtime import ModelRunner, Session, TensorInfo

sys.path.append(str(Path(__file__).parent.parent))
from enc_dec.run import TRTLLMEncDecModel
from llava import LlavaLlamaForCausalLM


DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument(
        "--visual_engine_dir",
        type=str,
        default=None,
        help="Directory containing visual TRT engines",
    )
    parser.add_argument(
        "--llm_engine_dir",
        type=str,
        default=None,
        help="Directory containing TRT-LLM engines",
    )
    parser.add_argument(
        "--hf_model_dir", type=str, default=None, help="Directory containing tokenizer"
    )
    parser.add_argument(
        "--decoder_llm",
        action="store_true",
        help="Whether LLM is decoder-only or an encoder-decoder variant?",
    )
    parser.add_argument(
        "--blip_encoder",
        action="store_true",
        help="Whether visual encoder is a BLIP model",
    )
    parser.add_argument("--nougat", action="store_true", help="Run nougat pipeline")
    parser.add_argument(
        "--input_text",
        type=str,
        default="Question: which city is this? Answer:",
        help="Text prompt to LLM",
    )
    parser.add_argument(
        "--num_beams", type=int, help="Use beam search if num_beams >1", default=1
    )
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--image-file", type=str)

    return parser.parse_args()


class MultiModalModel:
    def __init__(self, args):
        self.args = args
        runtime_rank = tensorrt_llm.mpi_rank()
        device_id = runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.stream = torch.cuda.current_stream().cuda_stream

        self.init_tokenizer()
        self.init_llm()

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.hf_model_dir, use_fast=False
        )

        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

    def init_llm(self):
        self.model = ModelRunner.from_dir(
            self.args.llm_engine_dir, rank=tensorrt_llm.mpi_rank(), debug_mode=False
        )
        self.model_config = self.model.session._model_config

    def get_image_ready(self, image):
        self.visual_features, self.visual_atts = image_processing(image, self.args)

    def generate(self, pre_prompt, post_prompt, max_new_tokens):
        visual_features, visual_atts = self.visual_features, self.visual_atts
        pre_input_ids = self.tokenizer(
            pre_prompt, return_tensors="pt", padding=True
        ).input_ids.to("cuda")
        post_input_ids = self.tokenizer(
            post_prompt, return_tensors="pt", padding=True
        ).input_ids.to("cuda")
        length = pre_input_ids.shape[1] + post_input_ids.shape[1] + visual_atts.shape[1]
        input_lengths = torch.IntTensor([length]).to(torch.int32).to("cuda")

        input_ids, ptuning_args = self.setup_fake_prompts(
            visual_features, pre_input_ids, post_input_ids, input_lengths
        )

        prompt_table = ptuning_args[0]
        prompt_table = torch.stack([prompt_table])
        np.save("prompt_table.npy", torch_to_numpy(prompt_table))

        profiler.start("LLM")
        end_id = self.tokenizer.eos_token_id
        output_ids = self.model.generate(
            input_ids.to("cpu"),
            sampling_config=None,
            prompt_table_path="prompt_table.npy",
            max_new_tokens=max_new_tokens,
            end_id=end_id,
            pad_id=self.tokenizer.pad_token_id,
            top_k=self.args.top_k,
            num_beams=self.args.num_beams,
            output_sequence_lengths=False,
            return_dict=False,
        )

        profiler.stop("LLM")

        if tensorrt_llm.mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx] :],
                    skip_special_tokens=True,
                )
                for batch_idx in range(self.args.batch_size)
            ]
            stripped_text = [
                [
                    output_beams_list[batch_idx][beam_idx].strip()
                    for beam_idx in range(self.args.num_beams)
                ]
                for batch_idx in range(self.args.batch_size)
            ]
            return stripped_text
        else:
            return None

    def setup_fake_prompts(
        self, visual_features, pre_input_ids, post_input_ids, input_lengths
    ):
        # Assemble fake prompts which points to image embedding actually
        fake_prompt_id = torch.arange(
            self.model_config.vocab_size,
            self.model_config.vocab_size
            + visual_features.shape[0] * visual_features.shape[1],
            device="cuda",
        )
        fake_prompt_id = fake_prompt_id.reshape(
            visual_features.shape[0], visual_features.shape[1]
        )

        if post_input_ids is not None:
            input_ids = [pre_input_ids, fake_prompt_id, post_input_ids]
        else:
            input_ids = [fake_prompt_id, pre_input_ids]
        input_ids = torch.cat(input_ids, dim=1).contiguous().to(torch.int32).cuda()

        if self.args.decoder_llm or self.runtime_mapping.is_first_pp_rank():
            ptuning_args = self.ptuning_setup(visual_features, input_ids, input_lengths)
        else:
            ptuning_args = [None, None, None]

        return input_ids, ptuning_args

    def ptuning_setup(self, prompt_table, input_ids, input_lengths):
        if prompt_table is not None:
            task_vocab_size = torch.tensor(
                [prompt_table.shape[1]], dtype=torch.int32, device="cuda"
            )
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1], prompt_table.shape[2])
            )

            hidden_size = self.model_config.hidden_size
            if not self.args.decoder_llm:
                hidden_size *= self.runtime_mapping.tp_size
            assert (
                prompt_table.shape[1] == hidden_size
            ), "Prompt table dimensions do not match hidden size"

            prompt_table = prompt_table.cuda().to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(self.model_config.dtype)
            )
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if self.model_config.remove_input_padding:
            tasks = torch.zeros([torch.sum(input_lengths)], dtype=torch.int32).cuda()
            if args.decoder_llm:
                tasks = tasks.unsqueeze(0)
        else:
            tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]


def load_test_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def image_processing(images, args):
    """
    Return visual_features and visual_atts by given image
    """
    import json

    with open(args.hf_model_dir + "/config.json", "r") as jsonfile:
        config = json.load(jsonfile)
    vision_tower_name = getattr(
        config, "vision_tower", getattr(config, "mm_vision_tower", None)
    )
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name).to(
        device="cuda", dtype=torch.float16
    )
    with torch.no_grad():
        if type(images) is list:
            images = [
                image.unsqueeze(0) if len(image.shape) == 3 else image
                for image in images
            ]
            images = torch.cat(images, dim=0)
        dtype = next(vision_tower.parameters()).dtype

        image_forward_outs = vision_tower(images.to(dtype), output_hidden_states=True)
        if "mm_vision_select_layer" in config:
            select_hidden_state_layer = config["mm_vision_select_layer"]
        else:
            select_hidden_state_layer = -1
        if abs(select_hidden_state_layer) > 100:  # TOOD: find a better impl
            # -212 -> 12,
            idx1, idx2 = abs(select_hidden_state_layer) % 100, -(
                abs(select_hidden_state_layer) // 100
            )
            # print("selecting multiple indices", idx1, idx2)
            image_features = torch.cat(
                (
                    image_forward_outs.hidden_states[idx1],
                    image_forward_outs.hidden_states[idx2],
                ),
                dim=-1,
            )
        else:
            image_features = image_forward_outs.hidden_states[select_hidden_state_layer]
        if isinstance(vision_tower, CLIPVisionModel):  # clip case, not for sam
            image_features = image_features[:, 1:].to(images.dtype)  # (B, N, D)

    model = LlavaLlamaForCausalLM.from_pretrained(
        args.hf_model_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        use_cache=True,
    )
    mm_projector = model.model.mm_projector
    mm_projector.cuda()
    image_features = mm_projector(image_features)
    image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to("cuda")

    return image_features, image_atts


if __name__ == "__main__":
    args = parse_arguments()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tensorrt_llm.logger.set_level("info")
    runtime_rank = tensorrt_llm.mpi_rank()

    image = load_test_image(args.image_file)
    processor = AutoProcessor.from_pretrained(args.hf_model_dir)
    image = processor(text=args.input_text, images=image, return_tensors="pt")[
        "pixel_values"
    ]
    image = image.half().to("cuda")

    pre_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
    post_prompt = args.input_text + " ASSISTANT:"
    model = MultiModalModel(args)
    model.get_image_ready(image)
    num_iters = 10
    for _ in range(num_iters):
        stripped_text = model.generate(pre_prompt, post_prompt, args.max_new_tokens)

    if runtime_rank == 0:
        logger.info("---------------------------------------------------------")
        logger.info(f"\n[Q] {args.input_text}")
        logger.info(f"\n[A] {stripped_text}")
        logger.info(
            f'TensorRT-LLM LLM latency: {profiler.elapsed_time_in_sec("LLM") / num_iters} sec'
        )
        logger.info("---------------------------------------------------------")
