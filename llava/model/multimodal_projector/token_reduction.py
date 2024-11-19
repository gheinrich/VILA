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

import math
from typing import Callable, Tuple

from sklearn.cluster import KMeans
import torch
import torch.nn as nn


def do_nothing(x, mode=None):
        return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // k.

    Input size is [batch, tokens, channels].
    k indicates the stride for the first set.
    k = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """



    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int, h: int, sx: int, sy: int, r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> Tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomly in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.

    Code from https://github.com/dbolya/tomesd/tree/main/tomesd
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy*sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy*sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :] # src
        b_idx = rand_idx[:, :num_dst, :] # dst

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        if a.shape[1] < r:
            print(f"Warning: reducing {r} tokens to {a.shape[1]}")
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c), src=src)

        return out

    return merge, unmerge

def kmeans_pytorch(x, num_clusters, num_iters=10):
    # Step 1: Initialize centroids randomly from the input
    B, N, C = x.shape
    x_flat = x.view(B * N, C)

    # Randomly choose initial centroids from the input data
    centroids = x_flat[torch.randperm(B * N)[:num_clusters]]

    for i in range(num_iters):
        # Step 2: Compute distances between x_flat and centroids
        distances = torch.cdist(x_flat, centroids)

        # Step 3: Assign each point to the nearest centroid
        cluster_indices = torch.argmin(distances, dim=1)

        # Step 4: Compute new centroids as the mean of points in each cluster
        centroids = torch.zeros(num_clusters, C).to(x.device, x.dtype)  # To accumulate new centroids
        for k in range(num_clusters):
            if (cluster_indices == k).sum() > 0:  # Avoid empty clusters
                centroids[k] = x_flat[cluster_indices == k].mean(dim=0)

    return centroids, cluster_indices


class ToME2D(nn.Module):
    """ToME token compression."""
    def __init__(self, w: int, h: int, sx: int, sy: int, r: int):
        super(ToME2D, self).__init__()
        self._w = w
        self._h = h
        self._sx = sx
        self._sy = sy
        self._r = r

    def forward(self, x):
        if isinstance(x, list):
            # Work out the expected number of remaining tokens.
            expected_tokens = self._w * self._h - self._r

            features = []

            sx = self._sx
            sy = self._sy

            for image in x:
                # Expect B,C,H,W.
                assert len(image.shape) == 4
                b, c, h, w = image.shape

                if w * h < expected_tokens:
                    previous_h, previous_w = h, w
                    if w < h:
                        w = math.ceil(expected_tokens / h)
                        # Pad the image to the right.
                        pad = (0, w - previous_w, 0, 0)
                    else:
                        h = math.ceil(expected_tokens / w)
                        # Pad the image to the bottom.
                        pad = (0, 0, 0, h - previous_h)
                    image = torch.nn.functional.pad(image, pad, value=0)
                    print(f"WARNING: ToME2D Padded image from h*w={previous_h}*{previous_w} to h*w={h}*{w} with pad={pad}")

                r = h * w - expected_tokens
                image = image.permute(0, 2, 3, 1).reshape(b, h*w, c)
                merge, _ = bipartite_soft_matching_random2d(image, w, h, sx, sy, r, no_rand=True)
                merged_x = merge(image)
                print(f"ToME2D (adaptive) input shape={image.shape} output_shape={merged_x.shape} r={r}")
                assert merged_x.shape == (b, expected_tokens, c), f"Expected {expected_tokens} tokens, got {merged_x.shape}"
                features.append(merged_x)

            merged_x = torch.cat(features, dim=0)
        else:
            merge, _ = bipartite_soft_matching_random2d(x, self._w, self._h, self._sx, self._sy, self._r, no_rand=True)
            merged_x = merge(x)
            print(f"ToME2D input shape={x.shape} output_shape={merged_x.shape}")
        return merged_x


class ToME(nn.Module):
    """ToME token compression."""
    def __init__(self, compression_ratio: int = 4):
        super(ToME, self).__init__()
        self._compression_ratio = compression_ratio

    def forward(self, x):
        merge, _ = kth_bipartite_soft_matching(x, k=self._compression_ratio)
        merged_x = merge(x)
        print(f"ToME input shape={x.shape} output_shape={merged_x.shape}")
        return merged_x

class KMeansCompression(nn.Module):
    """KMeans token compression/."""
    def __init__(self, compression_ratio: int = 4):
        super(KMeansCompression, self).__init__()
        self._compression_ratio = compression_ratio

    def forward(self, x):
        B, N, C = x.shape

        num_clusters = N // self._compression_ratio

        centroids, cluster_indices =  kmeans_pytorch(x, num_clusters=num_clusters)

        # Step 5: Use the cluster indices to compress the sequence
        # Reshape the cluster_indices to [B, N]
        cluster_indices = cluster_indices.view(B, N)

        # Choose one centroid for each cluster to represent the compressed sequence
        # We can use scatter_ or gather to aggregate information based on cluster indices
        compressed_x = torch.zeros(B, num_clusters, C).to(x.device, x.dtype)

        # Step 6: Aggregate the centroids based on cluster assignments
        for b in range(B):
            for k in range(num_clusters):
                # Find the tokens in sequence `b` that are assigned to cluster `k`
                mask = cluster_indices[b] == k
                if mask.sum() > 0:  # Avoid division by zero
                    compressed_x[b, k] = x[b, mask].mean(dim=0)

        print(f"KMeans input shape={x.shape} output_shape={compressed_x.shape}")
        return compressed_x
