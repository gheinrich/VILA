import os.path as osp
import unittest
from unittest.case import _id as __id
from unittest.case import skip as __skip

import numpy as np
import torch
import transformers
from parameterized import parameterized
from torch.utils.data import DataLoader
from transformers import SiglipImageProcessor

from llava import conversation as conversation_lib
from llava.data.dataset import make_supervised_data_module
from llava.train.args import DataArguments, TrainingArguments


def requires_lustre(reason=None):
    import os.path as osp

    if not (osp.isdir("/lustre") or osp.isdir("/mnt")):
        reason = "lustre path is not avaliable." if reason is None else reason
        return __skip(reason)
    return __id


DATASETS = [
    # "ccs_recaptioned"
    # "ccs_recaptioned_test",
    # "vflan",
    "coyo25m_0to10_vila15_40b_recap",
    # "coyo_25m",
    # "jukinmedia",
    # "panda70m",
    # "sharegpt4v_pretrain",
    # "sharegpt_video",
    # "sharegpt_video_qa",
    # "shot2story_shotonly",
    # "youcook2",
    # "video_chatgpt",
    # "vatex",
    # "internvid_1300K",
    # "internvid_10M",
    # "mmc4core",
]


def _test_fps_module(
    dataset_name,
    max_samples=-1,
    batch_size=32,
    num_workers=16,
    skip_before=0,
    num_video_frames=32,
    fps=2.0,
):
    # datasets_mixture.register_datasets_mixtures()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "lmsys/vicuna-7b-v1.5",
        model_max_length=8192 * 2,
        padding_side="right",
        use_fast=False,
        legacy=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    data_args = DataArguments(
        data_mixture=dataset_name,
        is_multimodal=True,
        lazy_preprocess=True,
        num_video_frames=num_video_frames,
        fps=fps,
    )
    data_args.image_processor = image_processor
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    training_args = TrainingArguments(
        output_dir="output",
    )

    data_args.mm_use_im_start_end = False
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    dataset = data_module["train_dataset"]

    dloader = DataLoader(
        dataset, collate_fn=data_module["data_collator"], batch_size=batch_size, num_workers=num_workers
    )
    dloader_len = len(dloader)
    len_list = []
    for idx, batch in enumerate(dloader):
        if idx < skip_before:
            continue

        if max_samples > 0 and idx > min(max_samples, dloader_len):
            break

        info = []
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                info.append((k, v.shape))
            else:
                info.append((k, type(v)))
            if k == "images":
                len_list.append(v.shape[0])
    print(f"[{idx}/{len(dloader)}]", info)
    # calculate the var of len_list
    print(f"len_list: {len_list}")
    print(f"var: {np.var(len_list)}")
    print(f"mean: {np.mean(len_list)}")
    assert np.var(len_list) > 0 and np.mean(len_list) > 0


def _test_make_supervised_data_module(
    dataset_name,
    max_samples=-1,
    batch_size=32,
    num_workers=16,
    skip_before=0,
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "lmsys/vicuna-7b-v1.5",
        # "Qwen/Qwen2.5-1.5B-Instruct",
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
        legacy=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    data_args = DataArguments(
        data_mixture=dataset_name,
        is_multimodal=True,
        lazy_preprocess=True,
    )
    data_args.image_processor = image_processor
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    training_args = TrainingArguments(
        output_dir="output",
    )

    data_args.mm_use_im_start_end = False
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    dataset = data_module["train_dataset"]

    dloader = DataLoader(
        dataset, collate_fn=data_module["data_collator"], batch_size=batch_size, num_workers=num_workers
    )
    dloader_len = len(dloader)
    for idx, batch in enumerate(dloader):
        if idx < skip_before:
            continue

        if max_samples > 0 and idx > min(max_samples, dloader_len):
            break

        info = []
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                info.append((k, v.shape))
            else:
                info.append((k, type(v)))
        print(f"[{idx}/{len(dloader)}]", info)


def main(dataset_name="coyo_25m_wds_spatial_ocr_bbox_interleaved_qas"):
    _test_make_supervised_data_module(dataset_name=dataset_name, batch_size=1, num_workers=0, max_samples=10)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
