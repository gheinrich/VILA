import os, os.path as osp, io
import argparse
import pprint
import pickle
from bisect import bisect
import base64
from PIL import Image
from filelock import Timeout, FileLock
import torch

from functools import lru_cache

@lru_cache(maxsize=16)
def load_pickle(pkl_path):
    return pickle.load(open(pkl_path, "rb"))


class SimpleCoyoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path="/home/jasonlu/vlm_datasets/debug/coyo-700m/pkl02-split",
        cache_dir="~/.cache/simplecoyo",
        overwrite_meta=False,
    ):
        self.data_path = data_path
        self.meta_path = osp.join(
            osp.expanduser(cache_dir), data_path.replace("/", "--") + ".meta.pkl"
        )
        
        self.pkl_list = sorted([f for f in os.listdir(data_path) if f.endswith(".pkl")])

        if osp.exists(self.meta_path) and not overwrite_meta:
            print(f"Loading meta information from {self.meta_path}")
            self.len_list, self.len_cumsum = load_pickle(self.meta_path)
        else:
            self.len_list = []
            self.len_cumsum = []

            print(f"Preparing meta information {self.data_path} ...")

            for idx, pkl_path in enumerate(self.pkl_list):
                print(f"analyze {idx}-of-{len(self.pkl_list)}, {pkl_path}")
                pkl = load_pickle(osp.join(data_path, pkl_path))
                self.len_list.append(len(pkl))
                self.len_cumsum.append(sum(self.len_list))

            print(f"Saving meta information to {self.meta_path}")
            os.makedirs(osp.dirname(self.meta_path), exist_ok=True)
            with FileLock(self.meta_path.replace(".meta.pkl", ".lock")):
                pickle.dump(
                    (self.len_list, self.len_cumsum), open(self.meta_path, "wb")
                )

    def __getitem__(self, idx):
        pkl_idx = bisect(self.len_cumsum, idx)
        innert_idx = idx - self.len_cumsum[pkl_idx] + self.len_list[pkl_idx]
        # print(pkl_idx, innert_idx)
        pkl_path = self.pkl_list[pkl_idx]
        pkl = load_pickle(osp.join(self.data_path, pkl_path))

        data = pkl[innert_idx]
        rawbytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        return dict(
            image=rawbytes,
            id=data["id"],
            url=data["url"],
            text=data["text"],
        )

    def __len__(self):
        return self.len_cumsum[-1]


if __name__ == "__main__":
    import torch
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    dist.init_process_group(
        backend="nccl",
    )

    train_dataset = SimpleCoyoDataset()
    # print(train_dataset[0])
    # print(train_dataset[12440])
    # print(train_dataset[12450])
    # print(train_dataset[124410])
    # exit(0)
    sampler = DistributedSampler(train_dataset)
    # sampler = None
    dloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler)
    sampler.set_epoch(0)
    print(len(train_dataset), len(dloader))
    for idx, data in enumerate(dloader):
        print(idx, len(dloader))
