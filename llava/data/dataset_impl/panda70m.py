import base64
import copy
import io
import json
import logging
import os, os.path as osp
import pathlib
import pickle
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import Resize

import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                             IMAGE_TOKEN_INDEX)
from llava.data.dataset_impl.textocr import GenericDataset, preprocess_OCR
from llava.data.datasets_mixture import DATASETS
from llava.train.args import DataArguments, TrainingArguments

from io import BytesIO

DEFAULT_HIERTEXT = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/panda70m"
SPLIT = "panda70m_testing"


from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.decoder import DecoderType
from pytorchvideo.data.video import Video
from pytorchvideo.data.encoded_video import select_video_class, EncodedVideo

class VILAEncodedVideo(EncodedVideo):
    @classmethod
    def from_bytesio(
        cls, file_path: str , decode_audio: bool = True, decoder: str = "pyav"
    ):
        if isinstance(file_path, io.BytesIO):
            video_file = file_path
            file_path = "tmp.mp4"
        elif isinstance(file_path, str):
            # We read the file with PathManager so that we can read from remote uris.
            with g_pathmgr.open(file_path, "rb") as fh:
                video_file = io.BytesIO(fh.read())

        video_cls = select_video_class(decoder)
        return video_cls(video_file, pathlib.Path(file_path).name, decode_audio)

def load_video(video_path, jfino = None, num_video_frames = 8, image_size = 224,):
    # video_path = io.BytesIO(open(video_path, "rb").read())
    idx = 0
    timestamps = jfino["timestamp"][idx]
    caption = jfino["caption"][idx]
    
    begin_t, begin_s = str2time(timestamps[0])
    end_t, end_s = str2time(timestamps[1])
    try:
        video = VILAEncodedVideo.from_bytesio(video_path, decoder="decord", decode_audio=False)
        duration = float(video.duration)
        print("DEBUG", duration)
        assert duration >= 0.25
        video_outputs = video.get_clip(start_sec=begin_s, end_sec=end_s)["video"]
        assert video_outputs.size(1) > 8
        num_frames = video_outputs.shape[1]
        # step = (num_frames - 1) // 8 + 1
        step = num_frames // num_video_frames
        num_frames = num_frames - (num_frames % 8)
        indices = torch.floor(torch.arange(0, num_frames, step)).long()
        video_outputs = video_outputs[:, indices, :, :]
    except (FileNotFoundError, FileNotFoundError) as e:
        print(f"bad data path {video_path}")
        print(f"Error processing {video_path}: {e}")
        video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)

    c, b, h, w = video_outputs.size()
    image_tensor = torch.zeros(b, c, image_size, image_size, dtype=torch.uint8)
    video_frames = video_outputs.permute(1, 0, 2, 3).contiguous()
    video_frames = Resize(size=[image_size, image_size], antialias=True)(video_frames)
    image_tensor[:, :, :, :] = video_frames
    print(begin_s, end_s, caption)
    return image_tensor, caption


def str2time(s):
    t = datetime.strptime(s, "%H:%M:%S.%f")
    init = datetime.strptime("0:00:00.000", "%H:%M:%S.%f")
    return t, (t - init).total_seconds()

if __name__ == "__main__":
    from datetime import datetime
    video_path = osp.expanduser("~/nvr_elm_llm/dataset/panda70m/panda70m_testing/WxTjy7RY2yA.mp4")
    json_path = osp.expanduser("~/nvr_elm_llm/dataset/panda70m/panda70m_testing/WxTjy7RY2yA.json")

    # video_path = io.BytesIO(open(video_path, "rb").read())
    jinfo = json.load(open(json_path, "r"))
    img_t = load_video(video_path, jfino=jinfo)
    # print(img_t)
    # print(jinfo["timestamp"][0])
    
    s1 = datetime.strptime(jinfo["timestamp"][0][0], "%H:%M:%S.%f")
    s2 = datetime.strptime(jinfo["timestamp"][0][1], "%H:%M:%S.%f")
    print(type(s2-s1))
    print((s2 - s1).total_seconds())