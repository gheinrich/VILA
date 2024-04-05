from typing import List, Optional, Tuple

import torch
from torch import nn

import transformers
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaAttention
from einops import rearrange

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

from llava.train.sequence_parallel.globals import get_pg_manager, get_ulysess_sp_pg
from .hybrid_attn import HybridAttention
from .ulysses_attn import UlyssesAttention


def new_flash_attn_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    seqlens_in_batch=None,
):
    # if not self._flash_attn_uses_top_left_mask:
    #     causal = self.is_causal
    # else:
    #     causal = self.is_causal and query_length != 1
    causal = self.is_causal

    # Contains at least one padding token in the sequence

    # TODO (QH): Check the data
    # assert attention_mask is None
    assert causal is True

    attn_func = HybridAttention()
    attn_output = attn_func(
        query_states,
        key_states,
        value_states,
        dropout,
        softmax_scale,
        causal=causal,
    )

    return attn_output


def __init__(self, config: LlamaConfig):
    nn.Module.__init__(self)
    self.config = config
    self.attention_dropout = config.attention_dropout
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
            f" and `num_heads`: {self.num_heads})."
        )

    self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
    self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
    self._init_rope()

    # wrap two potential "local-attention" up with DeepSpeed Ulysses logic.
    self.ulysses_attn_varlen_func = UlyssesAttention(flash_attn_varlen_func, get_ulysess_sp_pg())
    self.ulysses_attn_func = UlyssesAttention(flash_attn_func, get_ulysess_sp_pg())


def _flash_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    seqlens_in_batch=None,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
    """
    # bs, seq, nh, hdim
    # print(f"Input shape to _flash_attention_forward: {query_states.shape}")
    # query_states = query_states.permute(1, 0, 2, 3).contiguous()
    # key_states = key_states.permute(1, 0, 2, 3).contiguous()
    # value_states = value_states.permute(1, 0, 2, 3).contiguous()
    # query_states = query_states[:, :1000, :, :]
    # key_states = key_states[:, :1000, :, :]
    # value_states = value_states[:, :1000, :, :]

    attn_output = self.ulysses_attn_func(
        query_states, key_states, value_states, dropout_p=dropout, softmax_scale=softmax_scale, causal=self.is_causal
    )
    # # reshape it back to b, s, nh, hdim
    # query_states = query_states.permute(1, 0, 2, 3).contiguous()
    # key_states = key_states.permute(1, 0, 2, 3).contiguous()
    # value_states = value_states.permute(1, 0, 2, 3).contiguous()
    # attn_output = attn_output.permute(1, 0, 2, 3).contiguous()
    return attn_output

    # # Contains at least one padding token in the sequence
    # if attention_mask is not None:
    #     # Shape: b, s, nh, hdim
    #     print(f"Input shape to _flash_attention_forward: {query_states.shape}")
    #     batch_size = query_states.shape[0]
    #     query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
    #         query_states, key_states, value_states, attention_mask, query_length, seqlens_in_batch=seqlens_in_batch
    #     )

    #     cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    #     max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

    #     # Reshape it to s, b, hh, hdim to use DeepSpeed
    #     # DL: Alternatively, we can also modify the permuting dimensions of DeepSpeed backend.
    #     # But this may decrease the portability of the code.

    #     print(
    #         f"Input shape after _upad_input: q {query_states.shape}, cu_seqlens_q {cu_seqlens_q}, max_seqlen_in_batch_q {max_seqlen_in_batch_q}"
    #     )

    #     query_states = query_states.permute(1, 0, 2, 3).contiguous()
    #     key_states = key_states.permute(1, 0, 2, 3).contiguous()
    #     value_states = value_states.permute(1, 0, 2, 3).contiguous()

    #     attn_output_unpad = self.ulysses_attn_varlen_func(
    #         query_states,
    #         key_states,
    #         value_states,
    #         cu_seqlens_q,
    #         cu_seqlens_k,
    #         max_seqlen_in_batch_q,
    #         max_seqlen_in_batch_k,
    #         dropout_p=dropout,
    #         softmax_scale=softmax_scale,
    #         causal=self.is_causal,
    #     )

    #     # reshape it back to b, s, nh, hdim
    #     query_states = query_states.permute(1, 0, 2, 3).contiguous()
    #     key_states = key_states.permute(1, 0, 2, 3).contiguous()
    #     value_states = value_states.permute(1, 0, 2, 3).contiguous()
    #     attn_output_unpad = attn_output_unpad.permute(1, 0, 2, 3).contiguous()

    #     attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    # else:
    #     query_states = query_states.permute(1, 0, 2, 3).contiguous()
    #     key_states = key_states.permute(1, 0, 2, 3).contiguous()
    #     value_states = value_states.permute(1, 0, 2, 3).contiguous()

    #     attn_output = self.ulysses_attn_func(
    #         query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
    #     )

    #     # reshape it back to b, s, nh, hdim
    #     query_states = query_states.permute(1, 0, 2, 3).contiguous()
    #     key_states = key_states.permute(1, 0, 2, 3).contiguous()
    #     value_states = value_states.permute(1, 0, 2, 3).contiguous()
    #     attn_output = attn_output.permute(1, 0, 2, 3).contiguous()

    # return attn_output


def new_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    seqlens_in_batch: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def apply_hybrid_attn_monkey_patch_llama():
    # transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward = new_flash_attn_forward

    # transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = new_decoder_forward
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ = __init__
    transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward = _flash_attention_forward
