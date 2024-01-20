import os, os.path as osp, io
import argparse
import pprint
import pickle
from bisect import bisect
import base64
from PIL import Image
import json
from filelock import Timeout, FileLock
from functools import lru_cache
import tarfile
from multiprocessing.pool import ThreadPool as Pool

import torch
import torch.distributed
from torch.utils.data import Dataset, get_worker_info, ConcatDataset


@lru_cache(maxsize=32)
def load_pickle(pkl_path):
    return pickle.load(open(pkl_path, "rb"))


@lru_cache(maxsize=32)
def load_tarfile(tar_path):
    # return pickle.load(open(tar_path, "rb"))
    return tarfile.open(tar_path)


class UnexpectedEOFTarFile(tarfile.TarFile):
    def _load(self):
        """Read through the entire archive file and look for readable
        members.
        """
        try:
            while True:
                tarinfo = self.next()
                if tarinfo is None:
                    break
        except tarfile.ReadError as e:
            assert e.args[0] == "unexpected end of data"
        self._loaded = True


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
            osp.expanduser(cache_dir), data_path.replace("/", "--") + f"--max_shards:{max_shards_to_load}" + ".meta.pkl"
        )
        self.max_shards_to_load = max_shards_to_load
        self.tar_list = sorted([f for f in os.listdir(data_path) if f.endswith(".tar")])

        def extract_tar_info(idx, data_path, tar_path):
            print(f"analyze {idx}-of-{len(self.tar_list)}, {tar_path}")
            tar = load_tarfile(osp.join(data_path, tar_path))

            img_files = [t for t in tar.getnames() if t.lower().endswith(".jpg")]
            json_files = [t for t in tar.getnames() if t.lower().endswith(".json")]
            txt_files = [t for t in tar.getnames() if t.lower().endswith(".txt")]
            id_list = [t[:-4] for t in tar.getnames() if t.lower().endswith(".txt")]

            return (
                img_files,
                json_files,
                txt_files,
                id_list,
            )

        if osp.exists(self.meta_path) and not overwrite_meta:
            print(f"Loading cached meta information from {self.meta_path}")
            self.id_list, self.len_list, self.len_cumsum = load_pickle(self.meta_path)
        else:
            if torch.distributed.is_initialized():
                print(f"Running in distributed but meta file {self.data_path} not founded.")
                print(f"Please run `python llava/train/simple_coyo_dataset.py` before launch the job.")
            else:
                print(f"Preparing meta information {self.data_path} ...")

            self.len_list = []
            self.len_cumsum = []
            self.id_list = []

            pool = Pool(32)
            jobs = []
            for idx, tar_path in enumerate(self.tar_list):
                jobs.append(
                    pool.apply_async(extract_tar_info, (idx, data_path, tar_path))
                )
                if self.max_shards_to_load is not None and idx > self.max_shards_to_load:
                    break
            pool.close()
            pool.join()

            for _ in jobs:
                img_files, json_files, txt_files, id_list = _.get()
                assert len(txt_files) == len(json_files)
                assert len(txt_files) == len(img_files)
                self.id_list.append(id_list)
                self.len_list.append(len(img_files))
                self.len_cumsum.append(sum(self.len_list))

            print(f"Saving meta information to {self.meta_path}")
            os.makedirs(osp.dirname(self.meta_path), exist_ok=True)
            with FileLock(self.meta_path.replace(".meta.pkl", ".lock")):
                pickle.dump(
                    (self.id_list, self.len_list, self.len_cumsum),
                    open(self.meta_path, "wb"),
                )

    def __getitem__(self, idx):
        tar_idx = bisect(self.len_cumsum, idx)
        innert_idx = idx - self.len_cumsum[tar_idx] + self.len_list[tar_idx]
        # print(pkl_idx, innert_idx)
        tar_path = self.tar_list[tar_idx]
        tar = load_tarfile(osp.join(self.data_path, tar_path))

        # data = tar[innert_idx]
        # rawbytes = base64.b64decode(data["image"])
        # image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        data_id = self.id_list[tar_idx][innert_idx]

        caption = tar.extractfile(f"{data_id}.txt").read().decode("utf-8")
        jsoninfo = json.loads(tar.extractfile(f"{data_id}.json").read().decode("utf-8"))
        rawbytes = tar.extractfile(f"{data_id}.jpg").read()
        image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

        return dict(
            image=image,
            id=data_id,
            url=jsoninfo["url"],
            text=caption,
        )

    def __len__(self):
        return self.len_cumsum[-1]

    @staticmethod
    def custom_collate(batch):
        batched_data = defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched_data[k].append(v)
        return dict(batched_data)

if __name__ == "__main__":
    import torch
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    train_dataset = SimpleCoyoDataset(
        max_shards_to_load=None
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
        print(idx, data, len(dloader))
        if idx > 1000:
            break
