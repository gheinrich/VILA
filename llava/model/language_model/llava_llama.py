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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import LlamaForCausalLM, LlamaConfig, PreTrainedModel, AutoConfig, AutoModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..multimodal_encoder.builder import build_vision_tower
from ..multimodal_projector.builder import build_mm_projector
from ..configuration_llava import LlavaConfig
from ..utils import get_model_config
from .builder import build_llm


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(PreTrainedModel, LlavaMetaModel, LlavaMetaForCausalLM):
    config_class = LlavaLlamaConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: LlavaLlamaConfig=None, *args, **kwargs) -> None:
        super().__init__(config)
        llm_cfg, vision_tower_cfg, mm_projector_cfg = get_model_config(config)
        
        self.llm = build_llm(
            llm_cfg, config, LlamaConfig, LlamaForCausalLM, *args, **kwargs
        )
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        self.post_config()

        assert (
            self.llm is not None
            or self.vision_tower is not None
            or self.mm_projector is not None
        ), "At least one of the components must be instantiated."

    ## TODO define generation method here

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
        # Note (kentang-mit@): we have a unit test for this function.
        if self.training:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            )
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            new_input_ids = input_ids
        # ed = time.time()
        # print(inputs_embeds.device, "preparation time:", ed - st, "s.")

        # st = time.time()
        outputs = self.llm.forward(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            seqlens_in_batch=sorted_seqlens_in_batch,
        )
        return outputs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            _inputs["images"] = images
        return _inputs

AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
