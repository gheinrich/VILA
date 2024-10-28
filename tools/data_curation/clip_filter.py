import argparse
import bisect
import json
import os
import os.path as osp
from glob import glob
from itertools import chain
from pprint import pprint
from typing import Any, Dict

import numpy as np
import torch
from datasets import load_dataset
from hydra.utils import instantiate
from PIL import Image
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, ProcessorMixin

from llava.data.builder import DATASETS, parse_mixture
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger
from llava.utils.media import extract_media


def main(
    topk_p=30,
    output_dir="data_curation_dev/val_siglip",
    indices_folder="data_curation_dev/filter_index@1023_by_mmmu",
    benchmark_target=None,
):
    mmdb_dir = "data_curation_dev/mmdb"
    features, text_features, metainfos = [], [], []
    # merge all text and image embeddings
    dataset_len_info = {
        "fpath": [],
        "name": [],
        "count": [],
    }
    sum_scores_list = []

    for fpath in glob(os.path.join(mmdb_dir, "*.jsonl")):
        print(f"[SFT] loading {fpath}")
        dataset_name = osp.splitext(osp.basename(fpath))[0]

        features.append(io.load(fpath.replace(".jsonl", ".pt"), map_location="cuda"))
        text_features.append(io.load(fpath.replace(".jsonl", "_text.pt"), map_location="cuda"))
        jsonl_info = io.load(fpath)
        metainfos.extend(jsonl_info)

        dataset_len_info["fpath"].append(fpath)
        dataset_len_info["name"].append(dataset_name)
        dataset_len_info["count"].append(len(jsonl_info))

    # find all filtered indices by benchmark-SFT scores
    # data_num = 5072012
    filter_data_num = None
    # filter_data_num = int(data_num * percent / 100.0) # keep top 75% most relavant data.
    indices_all = set()
    for fpath in glob(os.path.join(output_dir, "*.pth")):
        sum_score = io.load(fpath, map_location="cuda")
        """
        # 1 x #val_size
        """
        if benchmark_target is not None and not benchmark_target.lower() in fpath.lower():
            continue
        if filter_data_num is None:
            filter_data_num = int(sum_score.numel() * topk_p / 100.0)
        indices = sum_score.topk(filter_data_num, largest=True).indices
        print(f"[Benchmark] loading {fpath}", indices.shape, len(indices_all))
        indices_all.update(indices.tolist())

    indices_all = sorted(list(indices_all))

    output_folder = osp.join(indices_folder, str(topk_p))
    if benchmark_target is not None:
        output_folder = osp.join(f"{indices_folder}@{benchmark_target}", str(topk_p))
    os.makedirs(output_folder, exist_ok=True)

    filter_count = 0
    data_count = 0
    left_bound = 0
    total_info = {}
    for idx, name in enumerate(dataset_len_info["name"]):
        right_bound = left_bound + dataset_len_info["count"][idx]

        start_idx = bisect.bisect_left(indices_all, left_bound)
        end_idx = bisect.bisect_left(indices_all, right_bound)

        kept_index = [_ for _ in indices_all[start_idx:end_idx]]
        total_index = list(range(left_bound, right_bound))
        filtered_out_index = sorted([_ - left_bound for _ in (set(total_index) - set(kept_index))])

        print(
            name,
        )
        # print("\t", left_bound, right_bound)
        print(
            f"\t filtering {len(filtered_out_index)} of {dataset_len_info['count'][idx]}\t{(len(filtered_out_index) / dataset_len_info['count'][idx] * 100):.1f}%"
        )

        meta = {
            "total": dataset_len_info["count"][idx],
            "filter_count": len(filtered_out_index),
            "filter_percent": len(filtered_out_index) / dataset_len_info["count"][idx] * 100,
        }
        total_info[name] = meta
        with open(f"{output_folder}/{name}.json", "w") as f:
            json.dump(filtered_out_index, f, indent=2)
        left_bound = right_bound
        data_count += dataset_len_info["count"][idx]
        filter_count += len(filtered_out_index)
    total_info["_total"] = {
        "total": data_count,
        "filtered": filter_count,
        "filter_percent": filter_count / data_count * 100,
    }
    print(f"filtering {filter_count} of {data_count}, {filter_count/data_count:.2f}")
    with open(f"{output_folder}/_total.json", "w") as f:
        json.dump(total_info, f, indent=2)
    print("Saved to ", output_folder)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
