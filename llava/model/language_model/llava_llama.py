#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/


import os
from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.loss import soft_cross_entropy
from llava.model.utils.packing import set_seqlens_in_batch
from llava.utils import distributed as dist

from ...train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"


# FIXME we will follow the convention to add a new class for CausalLM in the future
class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        packing: bool = True,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        dpo_forward: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        if packing and self.training and not dpo_forward:
            if seqlens_in_batch is None:
                seqlens_in_batch = torch.sum(attention_mask, dim=1)
            set_seqlens_in_batch(seqlens_in_batch)

            (inputs_embeds, attention_mask, position_ids, labels) = self.repack_multimodal_data(
                inputs_embeds, attention_mask, position_ids, labels
            )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs,
        )

        # Loss rescale for SP & DP loss match
        if dist.size() > 1:
            loss_weight = calculate_loss_weight(labels)
            outputs.loss = outputs.loss * loss_weight

        if dpo_forward:
            return outputs.logits, labels

        return outputs


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
