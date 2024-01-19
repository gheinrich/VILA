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

import wids

# @lru_cache(maxsize=32)
def load_tarfile(tar_path):
    return tarfile.open(tar_path)


class SimpleCoyoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-25m-vila",
        cache_dir="~/.cache/simplecoyo",
        overwrite_meta=False,
        image_load_mode="pil",  # pil / rawbytes / fpath,
        max_shards_to_load = None,
    ):
        self.data_path = data_path
        self.meta_path = osp.join(
            osp.expanduser(cache_dir), data_path.replace("/", "--") + f".max_shards:{max_shards_to_load}" + ".wdsmeta.json"
        )
        self.max_shards_to_load = max_shards_to_load
        
        if not osp.exists(self.meta_path):
            print(f"Walking through dirs {data_path}")
            # tar_list = sorted([f for f in os.listdir(data_path) if f.endswith(".tar")])
            tar_list = []
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    fpath = osp.join(root, file)
                    fpath = osp.relpath(fpath, data_path)
                    if not fpath.endswith(".tar"):
                        continue
                    tar_list.append(fpath)
            tar_list = sorted(tar_list)
            
            meta = {
                "name": "coyo-dev",
                "__kind__": "SimpleCoyoDataset",
                "wids_version": 1,
                "shardlist": [],
            }
            for idx, tar_relpath in enumerate(tar_list):
                tar_abspath = osp.join(data_path, tar_relpath)
                tar_meta_path = osp.join(
                    osp.expanduser(cache_dir), "dev", tar_abspath.replace("/", "--") + ".wdsmeta.json"
                )
                print(f"Fetch meta information {tar_abspath} ... {idx}-of-{len(tar_list)}")
                if not osp.exists(tar_meta_path):
                    print(f"    Generating meta: {tar_meta_path}")
                    tar = load_tarfile(tar_abspath)
                    uuids = list(set([".".join(_.split(".")[:-1]) for _ in tar.getnames()]))
                    nsamples = len(uuids)
                    url = osp.realpath(tar_abspath)
                    tar_meta = {
                        "url": url,
                        "nsamples": nsamples,
                        "filesize": osp.getsize(tar_abspath)
                    }
                    os.makedirs(osp.dirname(tar_meta_path), exist_ok=True)
                    json.dump(tar_meta, open(tar_meta_path, "w+"), indent=2)
                
                tar_meta = json.load(open(tar_meta_path, "r"))
                meta["shardlist"].append(tar_meta)
                if self.max_shards_to_load is not None and idx > self.max_shards_to_load:
                    break
            
            os.makedirs(osp.dirname(self.meta_path), exist_ok=True)
            json.dump(meta, open(self.meta_path, "w+"), indent=2)
        
        print(f"Loading webdataset from meta {self.meta_path}", flush=True)
        self.dataset = wids.ShardListDataset(self.meta_path)

        
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
                    a[k] = [v, ]
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
                        raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
                    c[key] = a[key] if key in a else b[key]
            return c
        
        
        tasks = (transform2list(_) for _ in batch)
        return reduce(merge, tasks)



if __name__ == "__main__":
    import torch
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    data_path="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata"
    # obelisc
    # data_path = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/interleaved/obelisc/stage4/no-partial"
    train_dataset = SimpleCoyoDataset(
        data_path=data_path,
        max_shards_to_load=None,
    )

    sampler = None
    from PIL import Image
    from torch.utils.data import default_collate
    from collections import defaultdict

    dloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        sampler=sampler,
        batch_size=2,
        collate_fn=SimpleCoyoDataset.custom_collate,
        num_workers=8,
    )
    # sampler.set_epoch(0)
    print(len(train_dataset), len(dloader))
    for idx, data in enumerate(dloader):
        print(f"{idx}-of-{len(dloader)}", data)
        if idx >= 10:
            break
