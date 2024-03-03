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
# This file is modified from https://github.com/haotian-liu/LLaVA/

from transformers import AutoConfig, PreTrainedModel
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from accelerate.hooks import add_hook_to_module


def is_mm_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    architectures = config.architectures
    for architecture in architectures:
        if "llava" in architecture.lower():
            return True
    return False


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print(
            "You are using newer LLaVA code base, while the checkpoint of v0 is from older code base."
        )
        print(
            "You must upgrade the checkpoint to the new code base (this can be done automatically)."
        )
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


def maybe_resize_pos_embeds(
    image_processor: BaseImageProcessor,
    model: PreTrainedModel,
    resolution: int = -1,
    interpolate_mode: str = "linear",
):
    if resolution in [model.config.image_size, -1]:
        return
    print(
        "You are resizing vision model's position embeddings to increase vision resolution..."
    )
    embeddings = model.vision_model.embeddings
    patch_size = embeddings.patch_size
    num_new_tokens = int((resolution // patch_size) ** 2)

    old_embeddings = embeddings.position_embedding
    match interpolate_mode:
        case "linear":
            import torch, torch.nn as nn

            if is_deepspeed_zero3_enabled():
                import deepspeed

                with deepspeed.zero.GatheredParameters(
                    [old_embeddings.weight], modifier_rank=None
                ):
                    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
            else:
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
            new_embeddings = nn.Embedding(
                num_new_tokens,
                old_embedding_dim,
                dtype=old_embeddings.weight.dtype,
                device=old_embeddings.weight.device,
            )
            ## Interpolate position embeddings
            mapped_indices = (
                torch.arange(num_new_tokens).to(old_embeddings.weight.device)
                / (num_new_tokens - 1)
                * (old_num_tokens - 1)
            )
            floor_indices = torch.clamp(
                mapped_indices.floor().long(), min=0, max=old_num_tokens - 1
            )
            ceil_indices = torch.clamp(
                mapped_indices.ceil().long(), min=0, max=old_num_tokens - 1
            )
            if is_deepspeed_zero3_enabled():
                params = [old_embeddings.weight, new_embeddings.weight]
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    interpolated_embeds = (mapped_indices - floor_indices)[
                        :, None
                    ] * old_embeddings.weight.data[floor_indices, :] + (
                        ceil_indices - mapped_indices
                    )[
                        :, None
                    ] * old_embeddings.weight.data[
                        ceil_indices, :
                    ]
            else:
                interpolated_embeds = (mapped_indices - floor_indices)[
                    :, None
                ] * old_embeddings.weight.data[floor_indices, :] + (
                    ceil_indices - mapped_indices
                )[
                    :, None
                ] * old_embeddings.weight.data[
                    ceil_indices, :
                ]
            new_embeddings.weight.data = interpolated_embeds
        case _:
            raise NotImplementedError

    if hasattr(old_embeddings, "_hf_hook"):
        hook = old_embeddings._hf_hook
        add_hook_to_module(new_embeddings, hook)
    new_embeddings.requires_grad_(old_embeddings.weight.requires_grad)
    ## update vision encoder's configurations
    model.config.image_size = resolution
    if hasattr(image_processor, "crop_size"):
        # CLIP vision tower
        image_processor.crop_size = resolution
    else:
        # SIGLIP vision tower
        assert hasattr(image_processor, "size")
        image_processor.size = {"height": resolution, "width": resolution}
    ## TODO define a '_reinitialize' method for VisionTower
    embeddings.position_embedding = new_embeddings
    embeddings.image_size = resolution
    embeddings.num_patches = embeddings.num_positions = num_new_tokens
    embeddings.position_ids = (
        torch.arange(embeddings.num_positions)
        .expand((1, -1))
        .to(old_embeddings.weight.device)
    )
