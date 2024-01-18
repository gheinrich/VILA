import base64
import copy
import io
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import Resize
import decord
from decord import VideoReader



from llava import conversation as conversation_lib
from llava.train.token_config import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
)
from llava.train import datasets_mixture
from llava.train.dataset import LazySupervisedDataset, tokenizer_image_token


class LazyCoyoWebDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        multimodal_cfg: dict,
        n_samples_per_idx=4,
    ):
        super().__init__()

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        from nvgpt4.data import get_train_dataset, get_loader, CaptioningWebdataset
        from nvgpt4.data import get_loader, get_train_dataset, WorkerConfig
        
        path = data_path
        # "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/captioning/coyo-700m_full_webdata"
        worker_config=WorkerConfig(
            rank=rank,
            world_size=world_size,
            num_workers=1,
        )

        print(f"Loading LazyCoyoWebDataset with rank:{rank} of world_size:{world_size}")
        s = time.time()
        train_dataset = get_train_dataset(
            path,
            batch_size=n_samples_per_idx,  
            image_decode="pil",
            worker_config=worker_config,
            shuffle_buffer_size=100, # 
            max_samples_per_sequence=None,
            # max_samples_per_sequence=100, # 100
            # virtual_epoch_length=1000 # 
        )
        print(f"Totally loading {len(train_dataset)} samples on rank {rank}-of-{world_size}")
        
        self.data_list = train_dataset
        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.data_list)

    def __getitem(self, i) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("")
    
    def __iter__(self) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        # info_list = self.data_list[i - self.idx_offset]
        info_list = next(self.data_list.__iter__())
        text_list = []
        image_list = []

        for sample in info_list:
            text_list.append(
                DEFAULT_IMAGE_TOKEN + sample.caption + self.tokenizer.eos_token
            )
            image = sample.image
            image_list.append(image)

        # following process is exactly same as training.
        image_list = torch.stack(
            [
                LazySupervisedDataset._process_image(image, self.multimodal_cfg)
                for image in image_list
            ]
        )

        # the same size for all images, so we concat
        cur_token_len = (
            image_list[0].shape[-2] // self.multimodal_cfg["patch_size"]
        ) * (image_list[0].shape[-1] // self.multimodal_cfg["patch_size"])
        cur_token_len += self.multimodal_cfg["n_extra_patch"]

        replace_token = DEFAULT_IMAGE_TOKEN
        if self.multimodal_cfg["use_im_start_end"]:
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
        text_list = [
            text.replace(DEFAULT_IMAGE_TOKEN, replace_token) for text in text_list
        ]

        if CONCAT_SAMPLES:
            # into <image>cap<eos><image>cap<eos>...
            text_list = "".join(text_list)

            # TODO: fix this to tokenize with images
            input_ids = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids  # 4, seq_len

            input_ids = input_ids[0]

        else:
            im_patch_token = self.tokenizer.convert_tokens_to_ids(
                [DEFAULT_IMAGE_PATCH_TOKEN]
            )[0]
            input_ids = [
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    n_image_tokens=cur_token_len,
                    image_token_index=im_patch_token,
                    return_tensors="pt",
                )
                for prompt in text_list
            ]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

        targets = input_ids.clone()
        im_patch_token = self.tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
        # mask image tokens
        targets[targets == im_patch_token] = IGNORE_INDEX
        # also mask start/end token
        if self.multimodal_cfg["use_im_start_end"]:
            im_start_token = self.tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN]
            )[0]
            im_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_END_TOKEN])[
                0
            ]
            targets[targets == im_start_token] = IGNORE_INDEX
            targets[targets == im_end_token] = IGNORE_INDEX
            assert (input_ids == im_start_token).sum() == (
                input_ids == im_end_token
            ).sum(), input_ids
        targets[targets == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=image_list)

