from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    apply_rotary_pos_emb,
)


def add_visual_attn_scale_to_llama(model):
    for m in model.modules():
        if isinstance(m, LlamaAttention):
            dev = next(m.parameters()).device
            dtype = next(m.parameters()).dtype
            print("Adding visual_attn_scale...")
            m.register_parameter(
                "visual_attn_scale",
                nn.Parameter(
                    (torch.zeros(m.num_heads) - 0.25).detach().to(dev).to(dtype)
                ),
            )


def new_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    import math

    bsz, q_len, ddd = hidden_states.size()

    if not hasattr(self, "q_proj_visual"):  # the original attention
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
    else:
        is_visual_feat = torch.zeros(bsz, q_len, dtype=torch.bool)  # [bsz, 1, q_len, 1]
        for i_sample, start, end in self.image_token_idx:
            is_visual_feat[i_sample, start:end] = 1
        is_visual_feat = is_visual_feat.to(hidden_states.device).view(-1)

        query_states = torch.zeros_like(hidden_states.view(-1, ddd))
        key_states = torch.zeros_like(hidden_states.view(-1, ddd))
        value_states = torch.zeros_like(hidden_states.view(-1, ddd))

        text_feat = hidden_states.view(-1, ddd)[~is_visual_feat, :]
        vis_feat = hidden_states.view(-1, ddd)[is_visual_feat, :]

        for layer_t, layer_v, out in [
            (self.q_proj, self.q_proj_visual, query_states),
            (self.k_proj, self.k_proj_visual, key_states),
            (self.v_proj, self.v_proj_visual, value_states),
        ]:
            text_out = layer_t(text_feat)
            vis_out = layer_v(vis_feat)
            out[~is_visual_feat, :] = text_out
            out[is_visual_feat, :] = vis_out

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    no_return_kv = False
    if past_key_value is not None:
        # reuse k, v, self_attention
        if (
            past_key_value[0].shape[0] != key_states.shape[0]
            and past_key_value[0].shape[0] == 1
        ):  # TODO: added support
            n_repeat = key_states.shape[0]
            past_key_value = (
                past_key_value[0].repeat(n_repeat, 1, 1, 1),
                past_key_value[1].repeat(n_repeat, 1, 1, 1),
            )
            no_return_kv = True
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (
        (key_states, value_states) if use_cache and not no_return_kv else None
    )

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights,
            torch.tensor(
                torch.finfo(attn_weights.dtype).min, device=attn_weights.device
            ),
        )

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def new_mlp_forward(self, x):
    # [b, n, d]
    b, n, d = x.shape
    is_visual_feat = torch.zeros(b, n, dtype=torch.bool)  # [bsz, 1, q_len, 1]
    if not self.training and not hasattr(
        self, "image_token_idx"
    ):  # during inference; pure text input
        pass
    else:
        for i_sample, start, end in self.image_token_idx:
            is_visual_feat[i_sample, start:end] = 1
    is_visual_feat = is_visual_feat.to(x.device).view(-1)

    out = torch.zeros_like(x.view(-1, d))

    text_feat = x.view(-1, d)[~is_visual_feat, :]
    vis_feat = x.view(-1, d)[is_visual_feat, :]

    text_feat = self.down_proj(
        self.act_fn(self.gate_proj(text_feat)) * self.up_proj(text_feat)
    )
    vis_feat = self.down_proj_visual(
        self.act_fn(self.gate_proj_visual(vis_feat)) * self.up_proj_visual(vis_feat)
    )

    out[~is_visual_feat, :] = text_feat
    out[is_visual_feat, :] = vis_feat
    out = out.reshape(b, n, d)

    return out


def new_mlp_forward_lora(self, x):
    # [b, n, d]
    b, n, d = x.shape
    is_visual_feat = torch.zeros(b, n, dtype=torch.bool)  # [bsz, 1, q_len, 1]
    for i_sample, start, end in self.image_token_idx:
        is_visual_feat[i_sample, start:end] = 1
    is_visual_feat = is_visual_feat.to(x.device).view(-1)

    x = x.view(-1, d)

    gate = self.gate_proj(x)
    up = self.up_proj(x)

    gate[is_visual_feat, :] = self.gate_proj_visual_lora(x[is_visual_feat, :])
    up[is_visual_feat, :] = self.up_proj_visual_lora(x[is_visual_feat, :])
    out = self.act_fn(gate) * up
    out = self.down_proj(out)
    out[is_visual_feat, :] = self.down_proj_visual_lora(out[is_visual_feat, :])
    out = out.reshape(b, n, d)
    return out


def create_lora_linear(dim, rank=128):
    lora = nn.Sequential(
        nn.Linear(dim, rank, bias=False),
        nn.Linear(rank, dim, bias=False),
    )
    import math

    nn.init.kaiming_uniform_(lora[0].weight, a=math.sqrt(5))
    nn.init.zeros_(lora[1].weight)
    return lora


def add_visual_expert_to_llama(
    model,
    add_visual_expert_mlp=False,
    add_visual_expert_attn=False,
    add_visual_lora=False,
):
    import copy

    if add_visual_expert_mlp:
        transformers.models.llama.modeling_llama.LlamaMLP.forward = new_mlp_forward

    if add_visual_lora:
        transformers.models.llama.modeling_llama.LlamaMLP.forward = new_mlp_forward_lora

    for m in model.modules():
        if add_visual_lora:
            if isinstance(m, LlamaAttention):
                if hasattr(m, "q_proj_visual_lora"):
                    print("LORA already added to attention... skip")
                    return
                print("Adding LORA to attention...")
                dtype, device = next(m.parameters()).dtype, next(m.parameters()).device
                m.q_proj_visual_lora = (
                    create_lora_linear(m.hidden_size).to(device).to(dtype)
                )
                m.k_proj_visual_lora = (
                    create_lora_linear(m.hidden_size).to(device).to(dtype)
                )
                m.v_proj_visual_lora = (
                    create_lora_linear(m.hidden_size).to(device).to(dtype)
                )
            if isinstance(m, LlamaMLP):
                if hasattr(m, "gate_proj_visual_lora"):
                    print("LORA already added to MLP... skip")
                    return
                print("Adding LORA to MLP...")
                dtype, device = next(m.parameters()).dtype, next(m.parameters()).device
                m.gate_proj_visual_lora = (
                    create_lora_linear(m.hidden_size).to(device).to(dtype)
                )
                m.up_proj_visual_lora = (
                    create_lora_linear(m.hidden_size).to(device).to(dtype)
                )
                m.down_proj_visual_lora = (
                    create_lora_linear(m.hidden_size).to(device).to(dtype)
                )

        if add_visual_expert_attn and isinstance(m, LlamaAttention):
            if hasattr(m, "q_proj_visual"):
                print("Visual expert already added to attention... skip")
                return
            print("Adding visual expert to attention...")
            m.q_proj_visual = copy.deepcopy(m.q_proj)
            m.k_proj_visual = copy.deepcopy(m.k_proj)
            m.v_proj_visual = copy.deepcopy(m.v_proj)

        if add_visual_expert_mlp and isinstance(m, LlamaMLP):
            if hasattr(m, "gate_proj_visual"):
                print("Visual expert already added to MLP... skip")
                return
            print("Adding visual expert to MLP...")
            m.gate_proj_visual = copy.deepcopy(m.gate_proj)
            m.up_proj_visual = copy.deepcopy(m.up_proj)
            m.down_proj_visual = copy.deepcopy(m.down_proj)
