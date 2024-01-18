# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,‚
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import pathlib
import re
import shutil
import time
from typing import Dict

import torch
import transformers

from llava import conversation as conversation_lib
from llava.model import *
from llava.train.arguments import DataArguments, ModelArguments, TrainingArguments
from llava.train.dataset import make_supervised_data_module
from llava.train.llava_trainer import LLaVATrainer
from llava.train.token_config import *
from llava.train import datasets_mixture

os.environ["WANDB_PROJECT"] = "llava"
PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def delete_partial_saved_ckpt(args):
    folder = args.output_dir
    if os.path.exists(folder):
        content = os.listdir(folder)
        checkpoints = [
            path
            for path in content
            if _re_checkpoint.search(path) is not None
        ]
        if len(checkpoints) > args.save_total_limit:
            if int(os.environ["RANK"]) == 0:
                latest_ckpt = os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))
                print(latest_ckpt)
                shutil.rmtree(latest_ckpt)
            else:
                while len(checkpoints) > args.save_total_limit:
                    time.sleep(3)
                    content = os.listdir(folder)
                    checkpoints = [
                        path
                        for path in content
                        if _re_checkpoint.search(path) is not None
                    ]


def train():
    datasets_mixture.register_datasets_mixtures()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    delete_partial_saved_ckpt(training_args)

    training_args.run_name = training_args.output_dir.split("/")[
        -1
    ]  # manually set wandb name
    if os.environ.get("NO_REPORT", False):
        training_args.report_to = []

    if ('Llama' in model_args.model_name_or_path or 
        'llama' in model_args.model_name_or_path or 
        'vicuna' in model_args.model_name_or_path or 
        'Vicuna' in model_args.model_name_or_path):
        model_cls = LlavaLlamaForCausalLM
    elif ('Mistral' in model_args.model_name_or_path or 
          'mistral' in model_args.model_name_or_path):
        model_cls = LlavaMistralForCausalLM
    elif ('Mixtral' in model_args.model_name_or_path or 
          'mixtral' in model_args.model_name_or_path):
        model_cls = LlavaMixtralForCausalLM

    if model_args.vision_tower is not None:
        # NOTE: a temporay hack to address the CPU OOM problem during model loading
        if "70" in model_args.model_name_or_path:
            import torch.distributed as dist

            rank = dist.get_rank()
            if rank % 8 >= 4:
                import time

                time.sleep(300)

        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            # low_cpu_mem_usage="70" in model_args.model_name_or_path,
        )
    else:
        raise ValueError('Vision Tower is None that is not expected.')

    if model_args.add_visual_attn_scale:
        print("Adding visual attention scale...")
        from llava.model.visual_attn_scale import add_visual_attn_scale_to_llama

        add_visual_attn_scale_to_llama(model)
        model.config.add_visual_attn_scale = True

    model_args.add_visual_expert = (
        model_args.add_visual_expert_mlp or model_args.add_visual_expert_attn
    )
    if model_args.add_visual_expert:
        # set the config, let the initialize function do the job
        model.config.add_visual_expert = model_args.add_visual_expert
        model.config.add_visual_expert_attn = model_args.add_visual_expert_attn
        model.config.add_visual_expert_mlp = model_args.add_visual_expert_mlp

    if model_args.neftune_alpha > 0:
        model.config.neftune_alpha = model_args.neftune_alpha

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=False,
    )

    if model_args.version == "v0":
        raise NotImplementedError("We only support Vicuna v1 models...")
    elif model_args.version == "vanilla":
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vanilla"
        ]
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1_1"
        ]

    # rewrite model_max_length in tokenizer to be the parameter set in training_args.
    tokenizer.model_max_length = training_args.model_max_length
    if model_args.vision_tower is not None:
        model.config.textual_embed_path = model_args.textual_embed_path
        model.config.min_max_range_path = model_args.min_max_range_path
        model_vision_dict = model.get_model().initialize_vision_modules(
            vision_tower=model_args.vision_tower,
            mm_vision_select_layer=model_args.mm_vision_select_layer,
            mm_projector_type=model_args.mm_projector_type,
            pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
            fsdp=training_args.fsdp,
        )
        model.get_vision_tower().to(dtype=torch.float16, device=training_args.device)
        vision_config = model_vision_dict["vision_config"]

        data_args.image_token_len = model_vision_dict["image_token_len"]
        data_args.image_processor = model_vision_dict["image_processor"]
        data_args.is_multimodal = True

        model.config.tune_mm_mlp_adapter = (
            training_args.tune_mm_mlp_adapter
        ) = model_args.tune_mm_mlp_adapter
        model.config.tune_vision_encoder = (
            training_args.tune_vision_encoder
        ) = model_args.tune_vision_encoder
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if model_args.tune_layer_norm:
                for m in model.modules():
                    if isinstance(m, torch.nn.LayerNorm):
                        for p in m.parameters():
                            p.requires_grad = True
            if model_args.tune_vision_encoder:
                print("* also tuning vision encoder...")
                for n, p in model.model.vision_tower.named_parameters():
                    p.requires_grad = True

        if model_args.tune_self_attn:
            model.requires_grad_(False)
            for n, p in model.named_parameters():
                if "self_attn" in n:
                    p.requires_grad = True

        if model_args.tune_ffn:
            model.requires_grad_(False)
            for n, p in model.named_parameters():
                if ".mlp." in n:
                    p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.freeze_self_attn = training_args.freeze_self_attn
        if training_args.freeze_self_attn:
            for n, p in model.named_parameters():
                if "self_attn" in n:
                    p.requires_grad = False

        # freeze the first half of the transformer layers
        model.config.freeze_first_half = training_args.freeze_first_half
        if training_args.freeze_first_half:
            n_layer = len(model.model.layers)
            for m in model.model.layers[: n_layer // 2]:
                for p in m.parameters():
                    p.requires_grad = False

        # freeze the later half of the transformer layers
        model.config.freeze_later_half = training_args.freeze_later_half
        if training_args.freeze_later_half:
            n_layer = len(model.model.layers)
            for m in model.model.layers[n_layer // 2 :]:
                for p in m.parameters():
                    p.requires_grad = False

        if training_args.train_visual_expert_only:
            print("*&" * 20)
            print("only updateing the visual expert...")
            for n, p in model.named_parameters():
                if "_visual" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        if training_args.projector_lr10:  # make sure we also tune this
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.mm_use_im_start_end = (
            data_args.mm_use_im_start_end
        ) = model_args.mm_use_im_start_end
        vision_config.use_im_start_end = (
            training_args.use_im_start_end
        ) = model_args.mm_use_im_start_end
        model.config.sep_image_conv_front = data_args.sep_image_conv_front

        model.initialize_vision_tokenizer(
            mm_use_im_start_end=model_args.mm_use_im_start_end,
            tokenizer=tokenizer,
            device=training_args.device,
            tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
            pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter,
        )

        if model_args.use_lora:
            print("Using LoRA tuning...")
            from peft import LoraConfig, TaskType, get_peft_model

            linear_layers = [
                n
                for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear)
                and "lm_head" not in n
                and "mm_projector" not in n
            ]
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=64,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=linear_layers,
            )
            model = get_peft_model(model, peft_config)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            model.print_trainable_parameters()

        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
        if len(params_no_grad) > 0 or training_args.projector_lr10:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print(
                        "[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}".format(
                            len(params_no_grad), params_no_grad
                        )
                    )
                else:
                    print(
                        "[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)".format(
                            len(params_no_grad), ", ".join(params_no_grad[:10])
                        )
                    )
                print(
                    "[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental."
                )
                print(
                    "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining"
                )

                from torch.distributed.fsdp.fully_sharded_data_parallel import (
                    FullyShardedDataParallel as FSDP,
                )

                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop("use_orig_params", True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)

                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    if "p32" in model_args.mm_projector_type:
        n_extra_patch = 32
    elif "se71" in model_args.mm_projector_type:
        n_extra_patch = 8
    elif "se" in model_args.mm_projector_type:
        n_extra_patch = 2
    elif "repeat" in model_args.mm_projector_type:
        n_extra_patch = 256
    else:
        n_extra_patch = 0

    if "sam" in str(type(model.get_vision_tower())).lower():
        patch_size = 16
    elif (
        "visiontransformer" in str(type(model.get_vision_tower())).lower()
        and model_args.mm_projector_type != "dsresampler"
        and "eva" not in str(type(model.get_vision_tower())).lower()
    ):
        patch_size = 28  # qwen
    elif "siglip" in str(type(model.get_vision_tower())).lower():
        if "16" in model_args.vision_tower:
            patch_size = 16
        elif "so400m" in model_args.vision_tower:
            patch_size = 14
        else:
            raise ValueError("Unknown siglip model, please set the patch size")
    else:  # clip or eva
        patch_size = 14
    patch_size = patch_size * 2 ** model_args.mm_projector_type.count("ds")

    data_module, extra_info = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        patch_size=patch_size,
        image_size=vision_config.image_size,
        n_extra_patch=n_extra_patch,
    )

    if training_args.projector_lr10:
        # from functools import partial
        LLaVATrainer.create_optimizer = LLaVATrainer.create_optimizer_fclr10

    # if (
    #     "mmc4" in data_args.dataset_type and data_args.aug_coyo
    # ):  # patch training sampler
    #     training_args.mmc4_samples = extra_info["mmc4_samples"]
    #     training_args.coyo_samples = extra_info["coyo_samples"]
    #     training_args.laion_samples = extra_info.get("laion_samples", None)
    #     LLaVATrainer._get_train_sampler = LLaVATrainer._get_local_train_sampler_aug_coyo
    # elif (
    #     "mmc4" in data_args.dataset_type
    #     or "wds" in data_args.dataset_type
    #     or "coyo" in data_args.dataset_type
    # ):  # patch training sampler
    #     LLaVATrainer._get_train_sampler = LLaVATrainer._get_local_train_sampler

    training_args.sample_lens = extra_info
    LLaVATrainer._get_train_sampler = LLaVATrainer._get_local_train_sampler_aug_coyo

    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
