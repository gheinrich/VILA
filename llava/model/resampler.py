import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return (
            F.interpolate(
                abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
                size=(tgt_size, tgt_size),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .flatten(0, 2)
            .to(dtype=dtype)
        )
    else:
        return abs_pos


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
        self,
        num_queries=144,
        embed_dim=4096,
        num_heads=32,
        kv_dim=1024,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Note: we remove pos embedding here

        self.query = nn.Parameter(torch.zeros(1, self.num_queries, embed_dim))
        trunc_normal_(self.query, std=0.02)

        self.kv_proj = nn.Linear(kv_dim, embed_dim * 2, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_q = norm_layer(embed_dim)
        self.ln_x = norm_layer(kv_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        org_x_shape = x.shape
        x = self.ln_x(x)
        x = self.kv_proj(x)
        k, v = torch.split(x, self.embed_dim, dim=-1)
        q = self.q_proj(self.ln_q(self.query))

        N = x.shape[0]

        out = self.attn(
            q.repeat(N, 1, 1),
            k,
            v,
            attn_mask=attn_mask,
        )[0]
        return out


class DSResampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
        self,
        grid_size=16,
        embed_dim=4096,
        num_heads=32,
        kv_dim=1664,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_queries = grid_size**2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Conv2d(kv_dim, embed_dim, 3, stride=2, padding=1)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        # x: NLD
        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        # reshapd and apply conv2d
        N, L, D = x.shape
        q = x.reshape(N, int(L**0.5), int(L**0.5), D).permute(0, 3, 1, 2)
        q = self.query(q).reshape(N, self.embed_dim, L // 4).permute(0, 2, 1)  # NLD
        q = self.ln_q(q).permute(1, 0, 2)  # LND

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)  # LND

        out = self.attn(
            q + self.pos_embed.unsqueeze(1).to(x.dtype),
            x + pos_embed.unsqueeze(1).to(x.dtype),
            x,
            attn_mask=attn_mask,
        )[0]
        out = out.permute(1, 0, 2)
        return self.ln_post(out)  # NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
