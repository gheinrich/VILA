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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
)
from transformers.models.siglip import (
    SiglipImageProcessor,
    SiglipVisionModel,
)
from transformers.models.mistral import (
    MistralConfig,
    MistralModel, 
    MistralForCausalLM
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
)

from llava.model.alt_llama_block import AltLlamaDecoderLayer
from llava.train.token_config import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)


class LlavaMistralConfig(MistralConfig):
    model_type = "llava_mistral"


class LlavaMistralModel(MistralModel):
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(LlavaMistralModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            if hasattr(config, "add_visual_expert") and config.add_visual_expert:
                print("Adding visual expert...")
                from llava.model.visual_attn_scale import add_visual_expert_to_llama

                add_visual_expert_to_llama(
                    self,
                    add_visual_expert_mlp=config.add_visual_expert_mlp,
                    add_visual_expert_attn=config.add_visual_expert_attn,
                )

            # HACK: for FSDP
            if self.vision_tower_class == "qwen":
                vision_tower = AutoModelForCausalLM.from_pretrained(
                    config.mm_vision_tower, trust_remote_code=True
                )
                vision_config = vision_tower.config
                vision_tower = vision_tower.transformer.visual
                if (
                    config.mm_projector_type == "dsresampler"
                ):  # remove the original resampler
                    vision_tower.proj = nn.Parameter(
                        torch.eye(vision_tower.ln_pre.bias.numel())
                        .to(vision_tower.proj.device)
                        .to(vision_tower.proj.dtype)
                    )
                    vision_tower.attn_pool = nn.Sequential()
                    vision_tower.ln_post = nn.Sequential()
                else:
                    vision_tower.proj = nn.Parameter(
                        torch.eye(vision_tower.proj.shape[-1])
                        .to(vision_tower.proj.device)
                        .to(vision_tower.proj.dtype)
                    )
                vision_config.image_size = vision_config.visual["image_size"]
                vision_config.patch_size = vision_config.visual["patch_size"]
                vision_config.hidden_size = vision_config.visual["output_dim"]
                vision_tower.config = vision_config
                self.vision_tower = [vision_tower]
            elif self.vision_tower_class == "eva":
                from llava.model.eva_clip import get_eva_clip_e

                self.vision_tower = [get_eva_clip_e(448).half()]
                self.vision_tower[0].config = CLIPVisionConfig(
                    **{
                        "hidden_size": 1792,
                        "image_size": 448,
                        "intermediate_size": 15360,
                        "model_type": "clip_vision_model",
                        "num_attention_heads": 16,
                        "num_channels": 3,
                        "num_hidden_layers": 64,
                        "patch_size": 14,
                    }
                )
            elif self.vision_tower_class == "siglip":
                self.vision_tower = [
                    SiglipVisionModel.from_pretrained(config.mm_vision_tower)
                ]
            else:
                self.vision_tower = [
                    CLIPVisionModel.from_pretrained(config.mm_vision_tower)
                ]
                # TODO: implement add visual expert here

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = self._get_mm_projector(
                config.mm_projector_type
                if hasattr(config, "mm_projector_type")
                else "linear",
                config.mm_hidden_size,
                config.hidden_size,
            )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    @property
    def vision_tower_class(self):
        if "qwen" in self.config.mm_vision_tower.lower():
            vision_tower_arch = "qwen"
        elif "eva" in self.config.mm_vision_tower.lower():
            vision_tower_arch = "eva"
        elif "raw" in self.config.mm_vision_tower.lower():
            vision_tower_arch = "raw"
        elif "siglip" in self.config.mm_vision_tower.lower():
            vision_tower_arch = "siglip"
        else:
            vision_tower_arch = "clip"
        return vision_tower_arch

    def initialize_vision_modules(
        self,
        vision_tower,
        mm_vision_select_layer,
        mm_projector_type,
        pretrain_mm_mlp_adapter=None,
        fsdp=None,
    ):
        self.config.mm_vision_tower = vision_tower

        # NOTE: should skip if we already have visual expert from pretrained weights
        if hasattr(self.config, "add_visual_expert") and self.config.add_visual_expert:
            print("Adding visual expert...")
            from llava.model.visual_attn_scale import add_visual_expert_to_llama

            add_visual_expert_to_llama(
                self,
                add_visual_expert_mlp=self.config.add_visual_expert_mlp,
                add_visual_expert_attn=self.config.add_visual_expert_attn,
            )
        if self.vision_tower_class == "siglip":
            image_processor = SiglipImageProcessor.from_pretrained(vision_tower)
        else:
            image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, "vision_tower"):
            if self.vision_tower_class == "qwen":
                vision_tower = AutoModelForCausalLM.from_pretrained(
                    vision_tower, trust_remote_code=True
                )
                vision_config = vision_tower.config
                vision_tower = vision_tower.transformer.visual
                if mm_projector_type == "dsresampler":  # remove the original resampler
                    vision_tower.proj = nn.Parameter(
                        torch.eye(vision_tower.ln_pre.bias.numel())
                        .to(vision_tower.proj.device)
                        .to(vision_tower.proj.dtype)
                    )
                    vision_tower.attn_pool = nn.Sequential()
                    vision_tower.ln_post = nn.Sequential()
                else:
                    vision_tower.proj = nn.Parameter(
                        torch.eye(vision_tower.proj.shape[-1])
                        .to(vision_tower.proj.device)
                        .to(vision_tower.proj.dtype)
                    )

                # vision_tower.proj = nn.Parameter(torch.eye(vision_tower.proj.shape[-1]).to(vision_tower.proj.device).to(vision_tower.proj.dtype))
                vision_config.image_size = vision_config.visual["image_size"]
                vision_config.patch_size = vision_config.visual["patch_size"]
                vision_config.hidden_size = vision_config.visual["output_dim"]
                vision_tower.config = vision_config
            elif self.vision_tower_class == "eva":
                from llava.model.eva_clip import get_eva_clip_e

                vision_tower = get_eva_clip_e(448).half()
                vision_config = vision_tower.config = CLIPVisionConfig(
                    **{
                        "hidden_size": 1792,
                        "image_size": 448,
                        "intermediate_size": 15360,
                        "model_type": "clip_vision_model",
                        "num_attention_heads": 16,
                        "num_channels": 3,
                        "num_hidden_layers": 64,
                        "patch_size": 14,
                    }
                )
            elif self.vision_tower_class == "raw":
                from llava.model.raw_vis_enc import RawVisionEncoder

                vision_tower = RawVisionEncoder(resolution=448, patch_size=14)
                vision_config = vision_tower.config = CLIPVisionConfig(
                    **{
                        "hidden_size": 3072,
                        "image_size": 448,
                        "model_type": "clip_vision_model",
                        "num_attention_heads": 16,
                        "num_channels": 3,
                        "num_hidden_layers": 64,
                        "patch_size": 14,
                    }
                )
            elif  'siglip' == self.vision_tower_class:
                vision_tower = SiglipVisionModel.from_pretrained(vision_tower)
                vision_config = vision_tower.config
            else:
                vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
                vision_config = vision_tower.config
        else:
            vision_tower = self.vision_tower[0]
            vision_config = vision_tower.config

        vision_tower.requires_grad_(False)

        # fsdp wrap
        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        # register the configs here...
        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_projector_type = mm_projector_type

        if not hasattr(self, "mm_projector"):
            self.mm_projector = self._get_mm_projector(
                mm_projector_type, vision_config.hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config,
        )

    def _get_mm_projector(self, projector_type, vision_hidden_size, llm_hidden_size):
        class TransformerN(nn.Module):
            def __init__(
                self, config, n_transformer, n_prompt_token=0, input_embeddings_avg=None
            ) -> None:
                super().__init__()

                dec_layer_cls = AltLlamaDecoderLayer

                self.layers = nn.ModuleList(
                    [
                        nn.Linear(vision_hidden_size, llm_hidden_size),
                        *[dec_layer_cls(config) for _ in range(n_transformer)],
                    ]
                )

                if n_prompt_token > 0:
                    self.prompt_token = nn.Parameter(
                        input_embeddings_avg.repeat(n_prompt_token, 1).unsqueeze(
                            0
                        ),  # 1, n_prompt_token, dim
                        requires_grad=True,
                    )
                else:
                    self.prompt_token = None

            def forward(self, x):
                for i, module in enumerate(self.layers):
                    x = module(x)
                    if isinstance(x, tuple):
                        assert len(x) == 1
                        x = x[0]
                    if i == 0 and self.prompt_token is not None:
                        x = torch.cat(
                            [self.prompt_token.repeat(x.shape[0], 1, 1), x], dim=1
                        ).contiguous()

                return x

        class Downsampler(nn.Module):
            def __init__(self, cin, cout, textual_centers=None) -> None:
                super().__init__()
                self.linear = nn.Linear(cin, cout)
                self.conv = nn.Conv2d(cout, cout, kernel_size=3, stride=2, padding=1)
                self.add_proj = textual_centers is not None
                if self.add_proj:
                    self.register_buffer("embed_center", textual_centers.detach())

            def forward(self, x):
                x = self.linear(x)
                x = torch.nn.GELU()(x)
                n, t, c = x.shape
                x = x.reshape(n, int(t**0.5), int(t**0.5), c).permute(0, 3, 1, 2)
                x = self.conv(x)
                out = x.reshape(n, c, -1).permute(0, 2, 1)

                if self.add_proj:
                    out = torch.softmax(out, dim=-1) @ self.embed_center.detach()
                return out

        class DownsampleConv(nn.Module):
            def __init__(self, cin, cout) -> None:
                super().__init__()
                self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                n, t, c = x.shape
                x = x.reshape(n, int(t**0.5), int(t**0.5), c).permute(0, 3, 1, 2)
                x = self.conv(x)
                out = x.reshape(n, x.shape[1], -1).permute(0, 2, 1)
                return out

        class StartEndAdder(nn.Module):
            def __init__(self, device, dtype, n_start=1, n_end=1) -> None:
                super().__init__()
                # NOTE: here we assume llm_hidden_size is equal to #centroids
                # may not apply in the future
                # initialize to take average
                ratio = 1 / (llm_hidden_size**0.5) * 10  # some manual alignment
                self.start_proj = (
                    nn.Parameter(
                        (torch.ones(1, n_start, llm_hidden_size) * ratio)
                        .to(device)
                        .to(dtype)
                    )
                    if n_start > 0
                    else None
                )
                self.end_proj = (
                    nn.Parameter(
                        (torch.ones(1, n_end, llm_hidden_size) * ratio)
                        .to(device)
                        .to(dtype)
                    )
                    if n_end > 0
                    else None
                )

            def forward(self, x):
                return torch.cat(
                    [
                        self.start_proj.repeat(x.shape[0], 1, 1),
                        x,
                        self.end_proj.repeat(x.shape[0], 1, 1),
                    ],
                    dim=1,
                )

        class TrailingToken(nn.Module):
            def __init__(self, avg_emb, n_tokens) -> None:
                super().__init__()
                self.n_tokens = n_tokens
                self.avg_emb = nn.Parameter(
                    avg_emb.view(1, 1, -1).repeat(1, n_tokens, 1).detach()
                )

            def forward(self, x):
                return torch.cat([x, self.avg_emb.repeat(x.shape[0], 1, 1)], dim=1)

        class TextualProjector(nn.Module):  # actually KNN projector
            def __init__(self, textual_centers, act="softmax") -> None:
                super().__init__()
                print("Using TextualProjector...")
                assert textual_centers is not None
                self.register_buffer("embed_center", textual_centers.detach())
                self.act = act

            def forward(self, x):
                if self.act == "softmax":
                    return torch.softmax(x, dim=-1) @ self.embed_center.detach()
                elif self.act == "sigmoid":
                    return (
                        torch.sigmoid(x)
                        @ self.embed_center.detach()
                        / (x.shape[-1] * 0.5)
                    )  # normed
                else:
                    raise NotImplementedError

        class RangeClip(nn.Module):  # actually KNN projector
            def __init__(self, min, max) -> None:
                super().__init__()
                self.register_buffer("min", min.detach().view(1, -1))
                self.register_buffer("max", max.detach().view(1, -1))

            def forward(self, x):
                # dimension broadcast auto done
                return torch.clamp(x, self.min.detach(), self.max.detach())

        class RangeMap(nn.Module):  # use sigmoid to interpolate between min and max
            def __init__(self, min, max) -> None:
                super().__init__()
                self.register_buffer("min", min.detach().view(1, -1))
                self.register_buffer("max", max.detach().view(1, -1))

            def forward(self, x):
                # dimension broadcast auto done
                if len(x.shape) == 2:
                    mmin = self.min.view(1, -1).detach()
                    mmax = self.max.view(1, -1).detach()
                elif len(x.shape) == 3:
                    mmin = self.min.view(1, 1, -1).detach()
                    mmax = self.max.view(1, 1, -1).detach()
                else:
                    raise NotImplementedError
                mmax = torch.max(mmax.abs(), mmin.abs())
                return torch.tanh(x) * mmax

        class Tuple2Tensor(nn.Module):
            def __init__(self, module) -> None:
                super().__init__()
                self.module = module

            def forward(self, x):
                return self.module(x)[0]

        class FlatNeighbor4(nn.Module):
            def forward(self, x):
                x_shape = list(x.shape)
                assert x_shape[-2] % 4 == 0
                x_shape[-2] = x_shape[-2] // 4
                x_shape[-1] = x_shape[-1] * 4
                return x.reshape(*x_shape)

        class FlatSquare4(nn.Module):
            def forward(self, x):
                # n, t, c
                n, t, c = x.shape
                assert t % 4 == 0
                reduced_hw = int(t**0.5) // 2
                x = x.reshape(n, reduced_hw, 2, reduced_hw, 2, c)
                x = x.permute(0, 1, 3, 2, 4, 5).reshape(n, reduced_hw**2, 4 * c)

                return x

        _embed = self.get_input_embeddings().weight
        if (
            hasattr(self.config, "textual_embed_path")
            and self.config.textual_embed_path is not None
        ):
            print(" * Loading textual embedding from", self.config.textual_embed_path)
            textual_centers = torch.load(self.config.textual_embed_path)
            textual_centers = textual_centers.to(_embed.device).to(_embed.dtype)
        else:
            textual_centers = None

        if (
            hasattr(self.config, "min_max_range_path")
            and self.config.min_max_range_path is not None
        ):
            print(" * Loading min_max_range_path from", self.config.min_max_range_path)
            min_max_range = torch.load(self.config.min_max_range_path)
            min_max_range = [
                r.to(_embed.device).to(_embed.dtype) for r in min_max_range
            ]
        else:
            min_max_range = None

        if projector_type == "linear":
            return nn.Linear(vision_hidden_size, llm_hidden_size)
        elif projector_type == "linear2":
            return nn.Linear(vision_hidden_size * 2, llm_hidden_size)
        elif projector_type == "linearproj":
            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                TextualProjector(textual_centers),
            )
        elif projector_type == "tf1":
            return TransformerN(self.config, 1)
        elif projector_type == "tf3":
            return TransformerN(self.config, 3)
        elif projector_type == "tf1p32":
            input_embeddings = self.get_input_embeddings().weight.data
            input_embeddings_avg = input_embeddings.mean(dim=0, keepdim=True).detach()
            return TransformerN(
                self.config,
                1,
                n_prompt_token=32,
                input_embeddings_avg=input_embeddings_avg,
            )
        elif projector_type == "tf1p32clip":
            assert min_max_range is not None

            input_embeddings = self.get_input_embeddings().weight.data
            input_embeddings_avg = input_embeddings.mean(dim=0, keepdim=True).detach()

            return nn.Sequential(
                TransformerN(
                    self.config,
                    1,
                    n_prompt_token=32,
                    input_embeddings_avg=input_embeddings_avg,
                ),
                RangeClip(*min_max_range),
            )
        elif projector_type == "tf1p32proj":
            assert min_max_range is not None

            input_embeddings = self.get_input_embeddings().weight.data
            input_embeddings_avg = input_embeddings.mean(dim=0, keepdim=True).detach()

            return nn.Sequential(
                TransformerN(
                    self.config,
                    1,
                    n_prompt_token=32,
                    input_embeddings_avg=input_embeddings_avg,
                ),
                nn.Linear(llm_hidden_size, 32000),
                TextualProjector(_embed.detach()),
            )
        elif projector_type == "downsample":
            return Downsampler(vision_hidden_size, llm_hidden_size)
        elif projector_type == "downsampleproj":
            return Downsampler(
                vision_hidden_size, llm_hidden_size, textual_centers=textual_centers
            )
        elif projector_type == "mlpse":
            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
                StartEndAdder(_embed.device, _embed.dtype),
            )
        elif projector_type == "mlpprojse":
            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
                StartEndAdder(_embed.device, _embed.dtype),
                TextualProjector(textual_centers),
            )
        elif projector_type == "mlpemb":
            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, min(32000, _embed.shape[0])),
                TextualProjector(_embed.detach()[:32000]),
            )
        elif projector_type == "downsampleprojse":
            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                DownsampleConv(llm_hidden_size, llm_hidden_size),
                StartEndAdder(_embed.device, _embed.dtype),
                TextualProjector(textual_centers),
            )
        elif projector_type == "dst1se":
            return nn.Sequential(
                DownsampleConv(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                Tuple2Tensor(LlamaDecoderLayer(self.config)),
                StartEndAdder(_embed.device, _embed.dtype),
                TextualProjector(textual_centers),
            )
        elif projector_type == "dsprojse71":
            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                DownsampleConv(llm_hidden_size, llm_hidden_size),
                StartEndAdder(_embed.device, _embed.dtype, n_start=7, n_end=1),
                TextualProjector(textual_centers),
            )
        elif projector_type == "linearsig":
            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                TextualProjector(textual_centers, act="sigmoid"),
            )
        elif projector_type == "linearclip":
            # load min, max range
            assert min_max_range is not None
            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                RangeClip(*min_max_range),
            )
        elif projector_type == "linearmap":
            # load min, max range
            assert min_max_range is not None
            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                RangeMap(*min_max_range),
            )

        elif projector_type == "linearrepeat":

            class Ch2Tok(nn.Module):
                def forward(self, x):
                    n, t, c = x.shape
                    return x.reshape(n, t * 2, c // 2)

            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size * 2), Ch2Tok()
            )

        elif projector_type == "dslinear":
            return nn.Sequential(
                FlatNeighbor4(),
                nn.Linear(vision_hidden_size * 4, llm_hidden_size),
            )
        elif projector_type == "dssqlinear":
            return nn.Sequential(
                FlatSquare4(),
                nn.Linear(vision_hidden_size * 4, llm_hidden_size),
            )
        elif projector_type == "dssqmlp":
            return nn.Sequential(
                FlatSquare4(),
                nn.Linear(vision_hidden_size * 4, llm_hidden_size * 2),
                nn.SiLU(),
                nn.Linear(llm_hidden_size * 2, llm_hidden_size),
            )
        elif projector_type == "dssqmlpmap":
            return nn.Sequential(
                FlatSquare4(),
                nn.Linear(vision_hidden_size * 4, llm_hidden_size * 2),
                nn.SiLU(),
                nn.Linear(llm_hidden_size * 2, llm_hidden_size),
                RangeMap(*min_max_range),
            )
        elif projector_type == "dssqlinearrepeat":

            class Repeat(nn.Module):
                def forward(self, x):
                    assert len(x.shape) == 3, x.shape  # n, t, c
                    return torch.cat((x, x), dim=1)

            return nn.Sequential(
                FlatSquare4(),
                nn.Linear(vision_hidden_size * 4, llm_hidden_size),
                Repeat(),
            )
        elif projector_type == "dssqlinearrepeat2":

            class RepeatProj(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.reshape = FlatSquare4()
                    self.linear1 = nn.Linear(vision_hidden_size * 4, llm_hidden_size)
                    self.linear2 = nn.Linear(vision_hidden_size * 4, llm_hidden_size)

                def forward(self, x):
                    x = self.reshape(x)
                    x1 = self.linear1(x)
                    x2 = self.linear2(x)
                    return torch.cat((x1, x2), dim=1)

            return RepeatProj()

        elif projector_type == "dssqlinearclip":
            return nn.Sequential(
                FlatSquare4(),
                nn.Linear(vision_hidden_size * 4, llm_hidden_size),
                RangeClip(*min_max_range),
            )
        elif projector_type == "dssqlinearp32":
            input_embeddings_avg = _embed.mean(dim=0, keepdim=True).detach()
            return nn.Sequential(
                FlatSquare4(),
                nn.Linear(vision_hidden_size * 4, llm_hidden_size),
                TrailingToken(input_embeddings_avg, 32),
                # RangeClip(*min_max_range),
            )

        elif projector_type == "dssqlinearp32clip":
            input_embeddings_avg = _embed.mean(dim=0, keepdim=True).detach()
            return nn.Sequential(
                FlatSquare4(),
                nn.Linear(vision_hidden_size * 4, llm_hidden_size),
                TrailingToken(input_embeddings_avg, 32),
                RangeClip(*min_max_range),
            )

        elif projector_type == "dsds":
            # class ProjDebugger(nn.Module):
            #     def forward(self, x):
            #         print(x.shape)
            #         exit()

            class Pre(nn.Module):
                def forward(self, x):
                    assert len(x.shape) == 4, x.shape  # n, h, w, c
                    return x.permute(0, 3, 1, 2)  # n, c, h, w

            class Post(nn.Module):
                def forward(self, x):
                    n, c, h, w = x.shape
                    return x.permute(0, 2, 3, 1).reshape(n, h * w, c)  # n, c, h, w

            return nn.Sequential(
                Pre(),
                nn.Conv2d(vision_hidden_size, llm_hidden_size, 3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(llm_hidden_size, llm_hidden_size, 3, stride=2, padding=1),
                nn.SiLU(),
                Post(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )

        elif projector_type == "dsresampler":
            from llava.model.resampler import DSResampler

            return nn.Sequential(
                DSResampler(),
                nn.Linear(4096, llm_hidden_size),
            )
        elif projector_type == "dsresampler144":
            from llava.model.resampler import Resampler

            return Resampler()

        elif projector_type == "mlp2x_gelu":
            return nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        else:
            raise NotImplementedError

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        seqlens_in_batch: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = self.get_vision_tower()

        from contextlib import nullcontext

        if (
            vision_tower is not None
            and (input_ids.shape[1] != 1 or self.training)
            and images is not None
        ):
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            with nullcontext() if getattr(
                self.config, "tune_vision_encoder", False
            ) else torch.no_grad():
                if type(images) is list:
                    images = [
                        image.unsqueeze(0) if len(image.shape) == 3 else image
                        for image in images
                    ]
                    images = torch.cat(images, dim=0)
                dtype = next(vision_tower.parameters()).dtype
                if "visiontransformer" in vision_tower.__class__.__name__.lower():
                    image_features = vision_tower(images.to(dtype))
                else:
                    image_forward_outs = vision_tower(
                        images.to(dtype), output_hidden_states=True
                    )
                    select_hidden_state_layer = getattr(
                        self.config, "mm_vision_select_layer", -1
                    )
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
                        image_features = image_forward_outs.hidden_states[
                            select_hidden_state_layer
                        ]
                if isinstance(vision_tower, CLIPVisionModel) or isinstance(vision_tower, SiglipVisionModel):  # clip case, not for sam
                    image_features = image_features[:, 1:].to(images.dtype)  # (B, N, D)
            image_features = self.mm_projector(image_features)

            if hasattr(self.config, "neftune_alpha") and self.config.neftune_alpha > 0:
                # print("using neftune tuning with alpha", self.config.neftune_alpha)
                dims = torch.tensor(image_features.shape[-2] * image_features.shape[-1])
                mag_norm = self.config.neftune_alpha / torch.sqrt(dims)
                image_features = image_features + torch.zeros_like(
                    image_features
                ).uniform_(-mag_norm, mag_norm)

            if self.config.mm_projector_type == "dsresampler":
                dummy_feat_shape = (1, 1024, 1664)
            elif self.config.mm_projector_type == "linear2":
                dummy_feat_shape = (1, 256, self.config.mm_hidden_size * 2)
            else:
                dummy_feat_shape = (1, 256, self.config.mm_hidden_size)

            dummy_image_features = torch.zeros(
                *dummy_feat_shape,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            dummy_image_features = self.mm_projector(dummy_image_features)[
                0
            ]  # (1, N, D)

            new_input_embeds = []
            cur_image_idx = 0

            image_token_idx = []

            num_patches = -1
            for i_sample, (cur_input_ids, cur_input_embeds) in enumerate(
                zip(input_ids, inputs_embeds)
            ):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = (
                        cur_input_embeds + (0.0 * dummy_image_features).sum()
                    )
                    new_input_embeds.append(cur_input_embeds)
                    #cur_image_idx += 1
                    continue
                if vision_tower.config.use_im_start_end:
                    # FIXME: these two lines are not used?
                    # TODO: not fixed for different image numbers yet
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (
                        cur_input_ids == vision_tower.config.im_end_token
                    ).sum():
                        print(
                            (cur_input_ids == vision_tower.config.im_start_token).sum(),
                            (cur_input_ids == vision_tower.config.im_end_token).sum(),
                        )
                        raise ValueError(
                            "The number of image start tokens and image end tokens should be the same."
                        )
                    image_start_tokens = torch.where(
                        cur_input_ids == vision_tower.config.im_start_token
                    )[0]
                    for image_start_token_pos in image_start_tokens:
                        if (
                            cur_image_idx >= image_features.shape[0]
                        ):  # SHOULD NOT HAPPEN!!!
                            if self.training:
                                print("%" * 20, "INDEXING ERROR!")
                                break
                            else:
                                raise ValueError("INDEXING ERROR!")
                        cur_image_features = image_features[cur_image_idx].to(
                            device=cur_input_embeds.device
                        )
                        num_patches = cur_image_features.shape[0]
                        if (
                            cur_input_ids[image_start_token_pos + num_patches + 1]
                            != vision_tower.config.im_end_token
                        ):
                            raise ValueError(
                                "The image end token should follow the image start token. "
                                + str(num_patches)
                            )
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:image_start_token_pos].detach(),
                                    cur_input_embeds[
                                        image_start_token_pos : image_start_token_pos
                                        + 1
                                    ],
                                    cur_image_features,
                                    cur_input_embeds[
                                        image_start_token_pos
                                        + num_patches
                                        + 1 : image_start_token_pos
                                        + num_patches
                                        + 2
                                    ],
                                    cur_input_embeds[
                                        image_start_token_pos + num_patches + 2 :
                                    ].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[: image_start_token_pos + 1],
                                    cur_image_features,
                                    cur_input_embeds[
                                        image_start_token_pos + num_patches + 1 :
                                    ],
                                ),
                                dim=0,
                            )
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    num_total_patches = (
                        (cur_input_ids == vision_tower.config.im_patch_token)
                        .sum()
                        .item()
                    )
                    masked_indices = torch.where(
                        cur_input_ids == vision_tower.config.im_patch_token
                    )[0]

                    while num_total_patches:
                        if (
                            cur_image_idx >= image_features.shape[0]
                        ):  # SHOULD NOT HAPPEN!!!
                            if self.training:
                                print("%" * 20, "INDEXING ERROR!")
                                break
                            else:
                                raise ValueError("INDEXING ERROR!")
                        cur_image_features = image_features[cur_image_idx]
                        num_patches = cur_image_features.shape[0]
                        mask_index_start = masked_indices[0]
                        masked_indices = masked_indices[num_patches:]

                        image_token_idx.append(
                            (
                                i_sample,
                                mask_index_start.item(),
                                (mask_index_start + num_patches).item(),
                            )
                        )

                        if orig_embeds_params is not None:
                            cur_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:mask_index_start].detach(),
                                    cur_image_features,
                                    cur_input_embeds[
                                        mask_index_start + num_patches :
                                    ].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:mask_index_start],
                                    cur_image_features,
                                    cur_input_embeds[mask_index_start + num_patches :],
                                ),
                                dim=0,
                            )
                        num_total_patches -= num_patches
                        assert num_total_patches >= 0, (num_total_patches, num_patches)
                        cur_image_idx += 1

                    new_input_embeds.append(cur_input_embeds)
                    if self.training:
                        if not masked_indices.numel() == 0:
                            print("%" * 20, "ERROR! masked_indices not empty...")
                    else:
                        assert masked_indices.numel() == 0

            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            if self.training:
                if not cur_image_idx == len(image_features):
                    print("%" * 20, f"ERROR! cur_image_idx {cur_image_idx} != len(image_feautres) {len(image_features)} ...")
            else:
                assert cur_image_idx == len(image_features), (
                    cur_image_idx,
                    len(image_features),
                )

            if (
                hasattr(self.config, "add_visual_attn_scale")
                and self.config.add_visual_attn_scale
                or hasattr(self.config, "add_visual_expert")
                and self.config.add_visual_expert
            ):
                for m in self.modules():
                    if isinstance(m, (LlamaAttention, LlamaMLP)):
                        m.image_token_idx = image_token_idx

            if (
                self.training
                and hasattr(self.config, "neftune")
                and self.config.neftune
            ):
                print("adding neftune...")
                # TODO:

        ret = super(LlavaMistralModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            seqlens_in_batch=seqlens_in_batch,
            position_ids=position_ids,
        )

        return ret


class LlavaMistralForCausalLM(MistralForCausalLM):
    config_class = LlavaMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = LlavaMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_vision_tower(self):
        model = self.get_model()
        vision_tower = model.vision_tower
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        seqlens_in_batch: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            seqlens_in_batch=seqlens_in_batch,
            position_ids=position_ids,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        seqlens_in_batch=None,
        position_ids=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "seqlens_in_batch": seqlens_in_batch,
                "position_ids": position_ids,
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(
        self,
        mm_use_im_start_end,
        tokenizer,
        device,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    ):
        vision_config = self.get_vision_tower().config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))
            (
                vision_config.im_start_token,
                vision_config.im_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
            )

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)
                ]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]


AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
