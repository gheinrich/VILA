import os, os.path as osp, io
import argparse
import pprint
import pickle
from bisect import bisect
import base64
from PIL import Image
import json
from filelock import Timeout, FileLock
from functools import lru_cache, reduce
import tarfile
from multiprocessing.pool import ThreadPool as Pool

import torch
import torch.distributed
from torch.utils.data import Dataset, get_worker_info, ConcatDataset

import getpass
from llava.wids import ShardListDataset


# @lru_cache(maxsize=32)
def load_tarfile(tar_path):
    return tarfile.open(tar_path)


COYO_25M_VILA = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila"
COYO_700M = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata"
COYO_700M_FILTERED = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata_fullmeta/stage2_filtered_v2"
OBELISC = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/interleaved/obelisc/stage4"
DATACOMP = "/lustre/fsw/portfolios/llmservice/users/dannyy/dannyy_gpt4/data_filtering/dc1b_filtered"


class SimpleCoyoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path=COYO_25M_VILA,
        # cache_dir="/home/ligengz/.cache/simplecoyo",
        # cache_dir="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/vila-webds-meta",
        cache_dir="/home/ligengz/datasets/vila-webds-meta",
        meta_path=None,
        image_load_mode="pil",  # pil / rawbytes / fpath,
        max_shards_to_load=None,
        overwrite=False,
    ):
        self.data_path = data_path
        self.meta_path = meta_path
        self.max_shards_to_load = max_shards_to_load

        _local_meta_path = osp.join(data_path, "wids-meta.json")
        print(_local_meta_path, osp.exists(_local_meta_path))
        if meta_path is None and osp.exists(_local_meta_path):
            self.meta_path = meta_path = _local_meta_path
            
        if meta_path is None :
            self.meta_path = osp.join(
                osp.expanduser(cache_dir),
                data_path.replace("/", "--")
                + f".max_shards:{max_shards_to_load}"
                + ".wdsmeta.json",
            )

        if not osp.exists(self.meta_path) or overwrite:
            # TODO(ligeng): speedup the generation
            #       1. parallelize the meta file generation 
            #       2. add options for meta file 
            assert (
                not torch.distributed.is_initialized()
            ), "Dataset meta file does not exist and generating may take a long time. \
                Please exit distributed mode and run `python llava/train/simple_coyo_dataset.py <webdataset path>`. \
                or set proper `meta_path=` when initializing."
            print(f"Meta path not found: {self.meta_path}")
            print(f"Walking through dirs {data_path}")
            # tar_list = sorted([f for f in os.listdir(data_path) if f.endswith(".tar")])
            tar_list = []
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    fpath = osp.join(root, file)
                    fpath = osp.relpath(fpath, data_path)
                    if not fpath.endswith(".tar"):
                        continue
                    # fpath = osp.abspath(osp.join(root, file))
                    tar_list.append(fpath)
            tar_list = sorted(tar_list)

            meta = {
                "name": "coyo-dev",
                "__kind__": "SimpleCoyoDataset",
                "wids_version": 1,
                "shardlist": [],
            }
            import shutil

            for idx, tar_relpath in enumerate(tar_list):
                tar_abspath = osp.join(data_path, tar_relpath)
                tar_meta_path = osp.join(
                    osp.expanduser(cache_dir),
                    "dev",
                    tar_abspath.replace("/", "--") + ".wdsmeta.json",
                )
                # print(data_path, tar_relpath, tar_abspath)
                # input()
                tar_realpath = osp.realpath(tar_abspath)
                tar_real_meta_path = osp.join(
                    osp.expanduser(cache_dir),
                    "dev",
                    tar_realpath.replace("/", "--") + ".wdsmeta.json",
                )

                print(
                    f"Fetch meta information {tar_abspath} ... {idx}-of-{len(tar_list)}"
                )
                
                if not osp.exists(tar_meta_path) and not osp.exists(tar_real_meta_path):
                    print(f"    Generating meta: {tar_meta_path}")
                    try:
                        tar = load_tarfile(tar_abspath)
                        uuids = list(
                            set([".".join(_.split(".")[:-1]) for _ in tar.getnames()])
                        )
                    except tarfile.ReadError as e:
                        print(f"Skipping {tar_abspath}")
                        print(e)
                        continue
                    nsamples = len(uuids)
                    url = osp.abspath(tar_abspath)
                    tar_meta = {
                        "url": url,
                        "nsamples": nsamples,
                        "filesize": osp.getsize(tar_abspath),
                    }
                    os.makedirs(osp.dirname(tar_meta_path), exist_ok=True)
                    json.dump(tar_meta, open(tar_meta_path, "w+"), indent=2)

                if osp.exists(tar_meta_path):
                    print(f"    Generating abs meta: {tar_meta_path}")
                    tar_meta = json.load(open(tar_meta_path, "r"))
                elif osp.exists(tar_real_meta_path):
                    print(f"    Generating abs meta: {tar_real_meta_path}")
                    tar_meta = json.load(open(tar_real_meta_path, "r"))
                else:
                    raise NotImplementedError

                tar_meta["url"] = osp.abspath(tar_abspath)
                # tar_meta["url"] = tar_relpath
                
                os.makedirs(osp.dirname(tar_meta_path), exist_ok=True)
                json.dump(tar_meta, open(tar_meta_path, "w+"), indent=2)
                if tar_meta_path != tar_real_meta_path and not osp.exists(tar_real_meta_path):
                    # tar_meta["url"] = osp.realpath(tar_abspath)
                    print(
                        f"    [abs2real] Copying {tar_meta_path} => {tar_real_meta_path}"
                    )
                    os.makedirs(osp.dirname(tar_real_meta_path), exist_ok=True)
                    json.dump(tar_meta, open(tar_real_meta_path, "w+"), indent=2)

                # input()
                meta["shardlist"].append(tar_meta)
                if (
                    self.max_shards_to_load is not None
                    and idx > self.max_shards_to_load
                ):
                    break
            # sorted by tar names
            meta["shardlist"] = sorted(meta["shardlist"], key=lambda x: x["url"])
            os.makedirs(osp.dirname(self.meta_path), exist_ok=True)
            json.dump(meta, open(self.meta_path, "w+"), indent=2)

        print(f"[SimplyCoyo] Loading meta infomation {self.meta_path}", flush=True)

        # uuid = abs(hash(self.meta_path)) % (10 ** 8)
        import hashlib

        uuid = hashlib.sha256(self.meta_path.encode()).hexdigest()[:8]
        self.dataset = ShardListDataset(
            self.meta_path,
            cache_dir=osp.expanduser(
                f"~/.cache/_wids_cache/{getpass.getuser()}-{uuid}"
            ),
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def simple_collate(batch):
        batched_data = {}
        for data in batch:
            for k, v in data.items():
                if k not in batched_data:
                    batched_data[k] = []
                batched_data[k].append(v)
        return dict(batched_data)

    @staticmethod
    def custom_collate(batch):
        def transform2list(a: dict):
            # trasnform all leaf nodes to list
            for k, v in a.items():
                if isinstance(v, dict):
                    a[k] = transform2list(v)
                else:
                    a[k] = [
                        v,
                    ]
            return a

        def merge(a: dict, b: dict, path=[], strict=False):
            c = {}
            keys = set(a.keys()).union(b.keys())
            # print(keys)
            for key in keys:
                if key in a and key in b:
                    if isinstance(a[key], dict) and isinstance(b[key], dict):
                        # apply recursively
                        c[key] = merge(a[key], b[key], path + [str(key)], strict=strict)
                    else:
                        c[key] = a[key] + b[key]
                else:
                    if strict:
                        raise Exception("Conflict at " + ".".join(path + [str(key)]))
                    c[key] = a[key] if key in a else b[key]
            return c

        tasks = (transform2list(_) for _ in batch)
        return reduce(merge, tasks)


if __name__ == "__main__":
    import torch
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", type=str, default=COYO_25M_VILA)
    # replaced by rank and world size
    parser.add_argument("-m", "--max-shards", type=int, default=None)
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()

    print("overwrite:", args.overwrite)
    train_dataset = SimpleCoyoDataset(
        data_path=args.data_path,
        max_shards_to_load=args.max_shards,
        # cache_dir="~/.cache/simplecoyo",
        overwrite=args.overwrite,
    )

    sampler = None
    from PIL import Image
    from torch.utils.data import default_collate
    from collections import defaultdict

    dloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        sampler=sampler,
        batch_size=1,
        collate_fn=SimpleCoyoDataset.custom_collate,
        # num_workers=8,
    )
    # sampler.set_epoch(0)
    print(len(train_dataset), len(dloader))
    for idx, data in enumerate(dloader):
        print(f"{idx}-of-{len(dloader)}", data)
        if idx >= 5:
            break