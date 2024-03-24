import base64
import copy
import glob
import io
import json
import logging
import os
import os.path as osp
import pathlib
import pickle
import random
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import PIL
# import transformers
from iopath.common.file_io import g_pathmgr
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import Resize

import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                             IMAGE_TOKEN_INDEX)
from llava.data.dataset import LazySupervisedDataset
from llava.data.dataset_impl.textocr import GenericDataset, preprocess_OCR
from llava.data.datasets_mixture import DATASETS
from llava.data.simple_vila_webdataset import VILAWebDataset
from llava.data.utils import VILAEncodedVideo
from llava.mm_utils import is_gemma_tokenizer, tokenizer_image_token
from llava.train.args import DataArguments, TrainingArguments

# DEFAULT_HIERTEXT = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/panda70m"
# SPLIT = "panda70m_testing"
print("import finish")

def str2time(s):
    t = datetime.strptime(s, "%H:%M:%S.%f")
    init = datetime.strptime("0:00:00.000", "%H:%M:%S.%f")
    return t, (t - init).total_seconds()


def load_video(video_path, jfino, idx=0, num_video_frames=8, image_size=334):
    import torch

    # video_path = io.BytesIO(open(video_path, "rb").read())
    timestamps = jfino["timestamp"][idx]
    caption = jfino["caption"][idx]

    begin_t, begin_s = str2time(timestamps[0])
    end_t, end_s = str2time(timestamps[1])
    try:
        video = VILAEncodedVideo.from_bytesio(video_path, decoder="decord", decode_audio=False)
        duration = float(video.duration)
        # print("DEBUG", duration)
        assert duration >= 0.25
        video_outputs = video.get_clip(start_sec=begin_s, end_sec=end_s)["video"]
        assert video_outputs.size(1) > num_video_frames
        num_frames = video_outputs.shape[1]
        # step = (num_frames - 1) // 8 + 1
        step = num_frames // num_video_frames
        num_frames = num_frames - (num_frames % 8)
        indices = torch.floor(torch.arange(0, num_frames, step)).long()
        video_outputs = video_outputs[:, indices, :, :]
    # except (FileNotFoundError, decord._ffi.base.DECORDError) as e:
    except Exception as e:
        print(f"bad data path {video_path}")
        print(f"Error processing {video_path}: {e}")
        video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)

    c, b, h, w = video_outputs.size()
    image_tensor = torch.zeros(b, c, image_size, image_size, dtype=torch.uint8)
    video_frames = video_outputs.permute(1, 0, 2, 3).contiguous()
    video_frames = Resize(size=[image_size, image_size], antialias=True)(video_frames)
    image_tensor[:, :, :, :] = video_frames
    # print(begin_s, end_s, caption)
    return image_tensor, caption, (begin_s, end_s)


class VILAPanda70m(Dataset):
    def __init__(
        self,
        data_path,
        image_folder,
        tokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ) -> None:
        super().__init__()

        data_path = osp.expanduser(data_path)
        # self.dataset = VILAWebDataset(data_path)
        self.dataset = VILAWebDataset(
            data_path="~/nvr_elm_llm/dataset/panda70m/webdataset",
            # meta_path="~/nvr_elm_llm/dataset/panda70m/webdataset/wids-mini.json",
        )

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_video_frames = data_args.num_video_frames if hasattr(data_args, "num_video_frames") else 8

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        video_path = data[".mp4"]
        jinfo = data[".json"]
        if "shortest_edge" in self.data_args.image_processor.size:
            image_size = self.data_args.image_processor.size["shortest_edge"]
        else:
            image_size = self.data_args.image_processor.size["height"]
        imgs, cap, secs = load_video(video_path, jfino=jinfo, image_size=image_size)
        # print(imgs.shape, cap, secs)
        num_video_frames = self.num_video_frames

        prompt = "<image>\n" * num_video_frames + cap
        # image_tensor = LazySupervisedDataset._load_video(video_path, num_video_frames, self.data_args)
        image_tensor = imgs
        processor = self.data_args.image_processor
        image_tensor = [
            processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in torch.unbind(image_tensor)
        ]
        image_tensor = torch.stack(image_tensor)

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )
        targets = copy.deepcopy(input_ids)
        data_dict = dict(input_ids=input_ids, labels=targets, image=image_tensor)

        return data_dict


def with_opencv(filename):
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return duration, fps, frame_count


def cleanup_corrupted_videos(
    workdir="~/nvr_elm_llm/dataset/panda70m/panda70m_training_2m",
    shards=0,
    total=-1,
):
    workdir = osp.expanduser(workdir)
    print("workdir", workdir)
    video_list = glob.glob(f"{workdir}/*.mp4")
    video_list = sorted(video_list)
    if total > 0:
        chunk = len(video_list) // total
        begin_idx = shards * chunk
        end_idx = (shards + 1) * chunk
        if shards == total - 1:
            end_idx = len(video_list)
        video_list = video_list[begin_idx:end_idx]
    print(f"checking total {len(video_list)} videos")
    # return 

    debug_info = {}
    for idx, video_path in enumerate(video_list):
        print(f"[{idx}/{len(video_list)}]", video_path)
        json_path = video_path.replace(".mp4", ".json")

        try:
            assert osp.exists(json_path) and osp.exists(video_path)
            jinfo = json.load(open(json_path, "r"))
            info = with_opencv(video_path)
            print(info)
            video = VILAEncodedVideo.from_bytesio(video_path, decoder="decord", decode_audio=False)
        except (RuntimeError, ZeroDivisionError) as e:
            debug_info[video_path] = str(e)
            print(f"!! deleting wrong [{idx}/{len(video_list)}]", video_path)  # , type(e))
            os.remove(video_path)
            os.remove(json_path)
            # input()
            time.sleep(3)

    print(debug_info)


if __name__ == "__main__":
    # WORKDIR=osp.expanduser("~/nvr_elm_llm/dataset/panda70m/panda70m_testing")
    # cleanup_corrupted_videos()
    import fire

    fire.Fire(cleanup_corrupted_videos)
    exit(0)

    # video_path = osp.expanduser(f"{WORKDIR}/WxTjy7RY2yA.mp4")
    # json_path = osp.expanduser(f"{WORKDIR}/WxTjy7RY2yA.json")
    jinfo = json.load(open(json_path, "r"))
    img_t = load_video(video_path, jfino=jinfo)
    print(img_t)

    # # print(jinfo["timestamp"][0])
    # s1 = datetime.strptime(jinfo["timestamp"][0][0], "%H:%M:%S.%f")
    # s2 = datetime.strptime(jinfo["timestamp"][0][1], "%H:%M:%S.%f")
    # print(type(s2-s1))
    # print((s2 - s1).total_seconds())

    # dst = VILAWebDataset(
    #     data_path="~/nvr_elm_llm/dataset/panda70m/webdataset",
    #     meta_path="~/nvr_elm_llm/dataset/panda70m/webdataset/wids-mini.json",
    # )

    # video_path = dst[0][".mp4"]
    # jinfo = dst[0][".json"]
    # img, cap, secs = load_video(video_path, jfino=jinfo)
    # print(img.shape, cap, secs)
