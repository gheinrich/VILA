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
import decord
import numpy as np
import PIL
import torch
import transformers
from decord._ffi.base import DECORDError
from iopath.common.file_io import g_pathmgr
from PIL import Image
from pytorchvideo.data.decoder import DecoderType
from pytorchvideo.data.encoded_video import EncodedVideo, select_video_class
from pytorchvideo.data.video import Video
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

DEFAULT_HIERTEXT = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/panda70m"
SPLIT = "panda70m_testing"


def with_opencv(filename):
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return duration, fps, frame_count


def split_video_to_clips(
    workdir=osp.expanduser("~/nvr_elm_llm/dataset/panda70m/panda70m_training_2m"),
    shards=0,
    total=-1,
):
    video_list = glob.glob(f"{workdir}/*.mp4")
    video_list = sorted(video_list)
    if total > 0:
        chunk = len(video_list) // total
        begin_idx = shards * chunk
        end_idx = (shards + 1) * chunk
        if shards == total - 1:
            end_idx = len(video_list)
        video_list = video_list[begin_idx:end_idx]
    print(f"Splitting total {len(video_list)} videos")
    output_dir = workdir + "_clip"
    debug_info = {}
    for idx, video_path in enumerate(video_list):
        print(f"[{idx}/{len(video_list)}]", video_path)
        json_path = video_path.replace(".mp4", ".json")
        assert osp.exists(json_path) and osp.exists(video_path)
        jinfo = json.load(open(json_path, "r"))
        print(jinfo)
        info = with_opencv(video_path)
        print(info)
        video = VILAEncodedVideo.from_bytesio(video_path, decoder="decord", decode_audio=False)

        return


if __name__ == "__main__":
    # WORKDIR=osp.expanduser("~/nvr_elm_llm/dataset/panda70m/panda70m_testing")
    # cleanup_corrupted_videos()
    import fire

    fire.Fire(split_video_to_clips)
