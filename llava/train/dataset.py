import base64
import copy
import io
import json
import logging
import os, os.path as osp
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
import torch
import transformers

from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import Resize
import decord
from decord import VideoReader
from io import BytesIO



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
from llava.train.utils import mprint, rprint

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def tokenizer_image_token(
    prompt, tokenizer, n_image_tokens=256, image_token_index=32000, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(
        prompt_chunks, [image_token_index] * (offset + n_image_tokens)
    ):
        input_ids.extend(x[offset:])

    # truncate to max length
    input_ids = input_ids[: tokenizer.model_max_length]

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def replace_image_patch_tokens(
    sources: Sequence[str],
    multimodal_cfg: dict,
) -> Dict:
    # NOTE: will NOT actually replace; just add start/end idx
    is_multimodal = multimodal_cfg["is_multimodal"]
    # image_token_len = multimodal_cfg['image_token_len']
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # swap the image to front
            if sentence["value"].endswith("\n" + DEFAULT_IMAGE_TOKEN):
                sentence["value"] = (
                    DEFAULT_IMAGE_TOKEN
                    + "\n"
                    + sentence["value"].replace("\n" + DEFAULT_IMAGE_TOKEN, "")
                ).rstrip()

            replace_token = DEFAULT_IMAGE_TOKEN
            if multimodal_cfg["use_im_start_end"]:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            ).rstrip()

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    n_image_tokens,
    conv_version=None,  # if non, use default...
    has_image=None,
) -> Dict:
    if conv_version is not None:
        conv = conversation_lib.conv_templates[conv_version].copy()
    else:  # goes to default
        conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    assert len(conversations) == 1

    if has_image is None:  # if not provided, try to guess
        has_image = DEFAULT_IMAGE_TOKEN in conversations[0]

    if has_image:
        im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        input_ids = torch.stack(
            [
                tokenizer_image_token(
                    prompt,
                    tokenizer,
                    n_image_tokens=n_image_tokens,
                    image_token_index=im_patch_token,
                    return_tensors="pt",
                )
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if total_len != target.numel():  # since we only have single sample here
            print("WARNING! Pad/Unknown token founded. Should not happen!!!")

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(
                    tokenizer_image_token(
                        rou,
                        tokenizer,
                        n_image_tokens=n_image_tokens,
                        image_token_index=im_patch_token,
                    )
                )
                instruction_len = (
                    len(
                        tokenizer_image_token(
                            parts[0],
                            tokenizer,
                            n_image_tokens=n_image_tokens,
                            image_token_index=im_patch_token,
                        )
                    )
                    - 2
                )
                if i>0:
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                if i>0:
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(sources)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored) "
                    f"{conversation}"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,  # note: attention mask generated in collector
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    n_image_tokens,
    conv_version=None,  # if None, use default...
    has_image=None,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    if conversation_lib.default_conversation.version == "v1":  # actually this one
        return preprocess_v1(
            sources,
            tokenizer,
            n_image_tokens=n_image_tokens,
            conv_version=conv_version,
            has_image=has_image,
        )
    else:
        raise NotImplementedError


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, "r") as f:
            list_data_dict = json.load(f)

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]

        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        multimodal_cfg: dict,
    ):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        with open(data_path, "r") as f:
            list_data_dict = json.load(f)

        print("total Dataset samples ", len(list_data_dict), " ", data_path)

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)

    @staticmethod
    def _process_image(image_file, multimodal_cfg: dict, resize=False):
        from torchvision import transforms

        image_folder = multimodal_cfg["image_folder"]
        processor = multimodal_cfg["image_processor"]
        image_size = multimodal_cfg["image_size"]
        if isinstance(image_file, str):
            if image_folder is not None:
                image_file = os.path.join(image_folder, image_file)
            image = Image.open(image_file).convert("RGB")
        elif isinstance(image_file, io.BytesIO):
            image = Image.open(image_file).convert("RGB")
        else:
            image = image_file  # already PIL image
        # special handling for 
        '''
        [4879083473105, 'https://cdn.billiger.com/dynimg/iBpF8x19A1EeE6JWhZ4CUgA2-pEoXYO2FO0obcY2xnQ1YO06rOi28g98iBnbjTFUopXq5ZfhHBQqF1VM8lIcu26sKkZG1CqYItu6E_XkUrRJATRZBfIhttOPYy5HiC-CEfUD0VilOp6Da-X9DPpbmdzQ7_-pwCreVTNv4QUAJ7hPqVE2WFUAuxagDi9LZMVqA/2061311384_large.png', 'AVM FRITZ!Repeater 1200 WLAN Mesh (866Mbit/s, 400Mbit/s), WLAN Repeater']
        '''
        h, w = image.size
        if h < 10 and w < 10:
            image = image.resize((30, 30))
            
        if resize:
            image = image.resize((image_size, image_size))
        if multimodal_cfg["image_aspect_ratio"] == "keep":
            max_hw, min_hw = max(image.size), min(image.size)
            aspect_ratio = max_hw / min_hw
            max_len, min_len = 448, 224
            shortest_edge = int(min(max_len / aspect_ratio, min_len))
            image = processor.preprocess(
                image,
                return_tensors="pt",
                do_center_crop=False,
                size={"shortest_edge": shortest_edge},
            )["pixel_values"][0]
        elif multimodal_cfg["image_aspect_ratio"] == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image

    def load_video(self, video_path, num_video_frames):
        decord.bridge.set_bridge("torch")
        video_reader = VideoReader(uri=video_path)
        image_size = self.multimodal_cfg["image_size"]
    
        idx = np.round(np.linspace(0, len(video_reader) - 1, num_video_frames)).astype(int)
        try:
            video_outputs = video_reader.get_batch(idx)
        except:
            print(f'bad data path {video_path}')
            video_outputs = torch.zeros(8, image_size, image_size, 3, dtype=torch.uint8)

        b, h, w, c = video_outputs.size()
        image_tensor = torch.zeros(b, c, image_size, image_size, dtype=torch.uint8)
        video_frames = video_outputs.permute(0, 3, 1, 2).contiguous()
        video_frames = Resize(size=[image_size, image_size], antialias=True)(video_frames)
        image_tensor[:, :, :, :] = video_frames

        return image_tensor


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sources = copy.deepcopy(sources)
        if "image" in sources[0]:
            if isinstance(sources[0]["image"], str):  # to list
                sources[0]["image"] = [sources[0]["image"]]
            image_file_list = sources[0]["image"]

            image = torch.stack(
                [
                    self._process_image(image_file, self.multimodal_cfg)
                    for image_file in image_file_list
                ]
            )

            # now random pick some context samples for training
            if self.multimodal_cfg["num_shots"] > 0:
                # TODO: currently we have different caption prompts (including \n location)
                # we need to check if it will affect the performance
                import numpy as np

                np.random.seed(i)
                query_set = np.random.choice(
                    len(self.list_data_dict),
                    self.multimodal_cfg["num_shots"],
                    replace=False,
                )
                context_images = [image]
                for context_i in query_set:
                    context_images.append(
                        self._process_image(
                            self.list_data_dict[context_i]["image"], self.multimodal_cfg
                        )
                    )
                    sources[0]["conversations"] += self.list_data_dict[context_i][
                        "conversations"
                    ].copy()
                image = torch.stack(context_images)

            # the same size for all images, so we concat
            cur_token_len = (image.shape[-2] // self.multimodal_cfg["patch_size"]) * (
                image.shape[-1] // self.multimodal_cfg["patch_size"]
            )
            cur_token_len += self.multimodal_cfg["n_extra_patch"]
            sources = replace_image_patch_tokens(
                [e["conversations"] for e in sources], self.multimodal_cfg
            )
        elif ("video" in sources[0]) or ("video_id" in sources[0]):
            num_video_frames = 8
            if "video" in sources[0]:
                video_file = sources[0]['video']
            else:
                video_file = sources[0]['video_id'] + '.mp4'
            video_folder = self.multimodal_cfg["image_folder"]
            video_path = os.path.join(video_folder, video_file)
            image_tensor = self.load_video(video_path, num_video_frames)
            processor = self.multimodal_cfg["image_processor"]
            
            image_tensor = [processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in torch.unbind(image_tensor)]
            image_tensor = torch.stack(image_tensor)

            if "video" in sources[0]:
                question = sources[0]['conversations'][0]["value"].rstrip()
                answer = sources[0]['conversations'][1]["value"].rstrip()
            else:
                question = sources[0]['q']
                answer = sources[0]['a']

            question = (
                question.replace("<image>\n", "")
                .replace("\n<image>", "")
                .replace("<image>", "")
            )
            question = (
                question.replace("<video>\n", "")
                .replace("\n<video>", "")
                .replace("<video>", "")
            )
            question = "<image>\n" * num_video_frames + question
            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

            cur_token_len = (image_tensor.shape[-2] // self.multimodal_cfg["patch_size"]) * (
                image_tensor.shape[-1] // self.multimodal_cfg["patch_size"]
            )
            cur_token_len += self.multimodal_cfg["n_extra_patch"]
            sources = replace_image_patch_tokens(
                [conversation], self.multimodal_cfg,
            )
        else:
            sources = [e["conversations"] for e in sources]
            cur_token_len = None

        data_dict = preprocess(
            sources,
            self.tokenizer,
            n_image_tokens=cur_token_len,
            has_image=cur_token_len is not None,
        )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
        elif self.multimodal_cfg["is_multimodal"]:
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.multimodal_cfg["image_processor"].crop_size
            # data_dict["image"] = torch.zeros(
            #     1, 3, crop_size["height"], crop_size["width"]
            # )
            data_dict["image"] = None

        return data_dict


class LazyWDSDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        multimodal_cfg: dict,
    ):
        super().__init__()
        n_samples = []
        n_shards = len(os.listdir(data_path)) // 3
        for shard in range(n_shards):
            with open(os.path.join(data_path, f"{shard:05d}_stats.json")) as f:
                info = json.load(f)
                n_samples.append(info["successes"])

        print("total samples", sum(n_samples))  # 10,881,869

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [
            sum(n_samples[i * shared_size : (i + 1) * shared_size])
            for i in range(world_size)
        ]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        tar_list = [
            f"{shard_idx:05d}.tar" for shard_idx in range(shard_start, shard_end)
        ]

        self.data_list = []
        t1 = time.time()
        for tar in tar_list:
            tmp_path = "/tmp/ccs{}".format(tar)
            tar_path = os.path.join(data_path, tar)

            os.makedirs(tmp_path, exist_ok=True)
            os.system(f"tar -xf {tar_path} -C {tmp_path}")

            txt_list = [f for f in os.listdir(tmp_path) if f.endswith(".txt")]

            for txt in txt_list:
                caption = open(os.path.join(tmp_path, txt), "r").read().strip()
                image_path = os.path.join(tmp_path, txt.split(".")[0] + ".jpg")
                self.data_list.append({"caption": caption, "image": image_path})
        t2 = time.time()
        print("Loading done. Total time: {:.2f} seconds".format(t2 - t1))

        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ADD_TEXT_PROMPT = False

        info = self.data_list[i - self.idx_offset]
        caption, image_path = info["caption"], info["image"]

        if ADD_TEXT_PROMPT:
            from llava.data.template import CAPTION_TEMPLATE

            rand_prompt = random.choice(CAPTION_TEMPLATE)
            rand_prompt = "<image>\n" + rand_prompt
        else:
            rand_prompt = "<image>\n"
        sources = [
            {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": rand_prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        ]

        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        if "image" in sources[0]:
            image = LazySupervisedDataset._process_image(
                sources[0]["image"], self.multimodal_cfg
            )
            image = torch.unsqueeze(image, dim=0)

            # now random pick some context samples for training
            if self.multimodal_cfg["num_shots"] > 0:
                raise NotImplementedError

            # the same size for all images, so we concat
            cur_token_len = (image.shape[-2] // self.multimodal_cfg["patch_size"]) * (
                image.shape[-1] // self.multimodal_cfg["patch_size"]
            )
            cur_token_len += self.multimodal_cfg["n_extra_patch"]
            sources = replace_image_patch_tokens(
                [e["conversations"] for e in sources], self.multimodal_cfg
            )
        else:
            raise NotImplementedError

        if not ADD_TEXT_PROMPT:
            assert len(sources) == 1
            # tokenize conversations
            image_tokens = tokenizer_image_token(
                sources[0][0]["value"],
                self.tokenizer,
                n_image_tokens=cur_token_len,
                return_tensors="pt",
            ).view(1, -1)
            text_tokens = self.tokenizer(
                [sources[0][1]["value"] + "</s>"],
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids
            input_ids = torch.cat([image_tokens, text_tokens[:, 1:]], dim=-1)
            targets = input_ids.clone()

            targets[:, : image_tokens.shape[-1]] = IGNORE_INDEX
            data_dict = dict(input_ids=input_ids, labels=targets)

        else:
            data_dict = preprocess(
                sources, self.tokenizer, n_image_tokens=cur_token_len
            )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        if image is not None:
            data_dict["image"] = image
        else:
            raise NotImplementedError
        # elif self.multimodal_cfg['is_multimodal']:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = self.multimodal_cfg['image_processor'].crop_size
        #     data_dict['image'] = torch.zeros(
        #         3, crop_size['height'], crop_size['width'])

        return data_dict


class LazyVFlanDataset(Dataset):
    """Dataset for supervised fine-tuning from flan mixture."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        multimodal_cfg: dict,
    ):
        super().__init__()
        import pickle

        self.list_data_dict = []

        logging.warning("Loading data...")
        pkl_list = os.listdir(data_path)
        for pkl in pkl_list:
            if pkl.endswith(".pkl"):
                with open(os.path.join(data_path, pkl), "rb") as f:
                    data = pickle.load(f)
                    self.list_data_dict.extend(data)
        logging.warning(f"Loaded {len(self.list_data_dict)} samples...")

        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data = self.list_data_dict[i]
        question = data["question"].rstrip()
        answer = data["answer:" if "answer:" in data else "answer"].rstrip()
        images = data["image:" if "image:" in data else "image"]

        if isinstance(images, str):
            images = [images]
        assert len(images) <= 8, "Too many images in one sample {}".format(len(images))
        # if len(images) == 8:  # sample it to be 4
        #     images = images[::2]
        n_images = len(images)

        decode_images = []
        for image_str in images:
            if image_str.endswith(".jpg"):
                decode_images.append(image_str)  # a path
            else:  # jpeg bytes
                rawbytes = base64.b64decode(image_str)
                decode_images.append(Image.open(io.BytesIO(rawbytes)).convert("RGB"))
        if n_images == 8:
            resize = True
        else:
            resize = False
        images = [
            LazySupervisedDataset._process_image(img, self.multimodal_cfg, resize=resize)
            for img in decode_images
        ]

        if self.multimodal_cfg["num_shots"] > 0:
            raise NotImplementedError  # do not support multi-shot for FLAN

        # let's make sure there is no <image> in the question...
        if (
            "Image Descriptions" in question
        ):  # NOTE: specicial handlement for generation_visual-dialog_train.pkl
            question_split = question.split("\nQuestion: ")[1:]
            qa_pairs = []
            for qa in question_split:
                qa_pairs.append(qa.split("\nAnswer: "))

            qa_pairs[0][0] = "<image>\n" + qa_pairs[0][0]
            assert len(qa_pairs[-1]) == 1
            qa_pairs[-1][0] = qa_pairs[-1][0].replace("\n", "")
            qa_pairs[-1].append(answer)
            conversation = []
            for q, a in qa_pairs:
                conversation.append({"from": "human", "value": q})
                conversation.append({"from": "gpt", "value": a})
        else:
            question = (
                question.replace("<image>\n", "")
                .replace("\n<image>", "")
                .replace("<image>", "")
            )
            question = "<image>\n" * n_images + question
            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

        # the same size for all images, so we concat
        if len(images) > 0:
            cur_token_len = (
                images[0].shape[-2] // self.multimodal_cfg["patch_size"]
            ) * (images[0].shape[-1] // self.multimodal_cfg["patch_size"])
            cur_token_len += self.multimodal_cfg["n_extra_patch"]
        else:
            assert not "<image>" in question
            cur_token_len = 0
        sources = replace_image_patch_tokens([conversation], self.multimodal_cfg)

        # NOTE: here we use the simple version without the system prompt
        if n_images == 8:
            conv_version = "vicuna_v1_1"
        else:
            conv_version = "vicuna_v1_1_nosys"
        data_dict = preprocess(
            sources,
            self.tokenizer,
            n_image_tokens=cur_token_len,
            conv_version=conv_version,
        )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        if len(images) > 0:
            data_dict["image"] = torch.stack(images)
        else:
            # crop_size = self.multimodal_cfg["image_processor"].crop_size
            # data_dict["image"] = torch.zeros(
            #     1, 3, crop_size["height"], crop_size["width"]
            # )
            # data_dict['image'] = torch.zeros(1, 3, 224, 224)
            data_dict["image"] = None

        return data_dict


class LazyMMC4Dataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        multimodal_cfg: dict,
        image_following_text_only=False,
        text_only=False,
    ):
        super().__init__()

        import pickle

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        count_info_list = sorted(
            [f for f in os.listdir(data_path) if f.endswith(".count")]
        )
        n_samples = [
            int(open(os.path.join(data_path, f), "r").read().strip())
            for f in count_info_list
        ]

        print("total MMC4 samples", sum(n_samples))  # 10,881,869

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [
            sum(n_samples[i * shared_size : (i + 1) * shared_size])
            for i in range(world_size)
        ]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                data_list = pickle.load(f)

            full_data_list.extend(data_list)

        print("* loaded totally {} samples".format(len(full_data_list)))

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg

        self.image_following_text_only = image_following_text_only
        self.text_only = text_only

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        info = self.data_list[i - self.idx_offset]

        sentences = info["text_list"]
        sim_matrix = info["similarity_matrix"]  # we do not use this...

        # convert images from base64 to PIL and filter based on image-text similarity
        images, sentence_ixs = [], []
        if not self.text_only:
            for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
                image_base64 = sample_image["image_base64"]
                rawbytes = base64.b64decode(image_base64)

                sim_ix = sample_image["matched_text_index"]
                # sim_ix = np.argmax(sim_vec)
                # sim_score = sim_vec[sim_ix]

                # filter to images >= 5KB
                # TODO: we should enable filtering here!!!
                # if len(rawbytes) // 1000 <= 5:
                #     continue
                # if sim_score < 0.24:
                #     continue
                image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

                images.append(image)
                sentence_ixs.append(sim_ix)

        # constrain max num 6 images
        max_num_images = 6
        if len(images) > max_num_images:
            images = images[:max_num_images]
            sentence_ixs = sentence_ixs[:max_num_images]

        # reorder images according to text insertion
        images = [images[iii] for iii in np.argsort(sentence_ixs)]

        # preprocess and tokenize text
        # TODO: do we need divide tokens here?  TODO: enable image start, end token!!!
        for ix in sentence_ixs:
            sentences[ix] = f"<image>{sentences[ix]}"

        if self.image_following_text_only:
            # use pad tokens to divide sentence pieces
            text = self.tokenizer.pad_token.join(sentences)
        else:
            text = " ".join(sentences)
        # whitespace cleanup
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        text = f"{text}{self.tokenizer.eos_token}"  # add eos token

        if len(images) > 0:
            images = torch.stack(
                [
                    LazySupervisedDataset._process_image(image, self.multimodal_cfg)
                    for image in images
                ]
            )

            # the same size for all images, so we concat
            cur_token_len = (
                images[0].shape[-2] // self.multimodal_cfg["patch_size"]
            ) * (images[0].shape[-1] // self.multimodal_cfg["patch_size"])
            cur_token_len += self.multimodal_cfg["n_extra_patch"]

            replace_token = DEFAULT_IMAGE_TOKEN
            if self.multimodal_cfg["use_im_start_end"]:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            text = text.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        else:
            images = None
            cur_token_len = 0

        im_patch_token = self.tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
        input_ids = tokenizer_image_token(
            text,
            self.tokenizer,
            n_image_tokens=cur_token_len,
            image_token_index=im_patch_token,
            return_tensors="pt",
        )
        assert len(input_ids.shape) == 1

        # now check the case where the last token is image patch token
        if input_ids[-1] == im_patch_token:  # need to remove one last image
            last_non_im_patch_indices = torch.where(input_ids != im_patch_token)[0][-1]
            if self.multimodal_cfg["use_im_start_end"]:  # will be strip
                assert (
                    input_ids[last_non_im_patch_indices]
                    == self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN])[0]
                )
            else:  # will be preserved
                last_non_im_patch_indices += 1
            input_ids = input_ids[:last_non_im_patch_indices]
        # now check if the last token is image start...
        if self.multimodal_cfg["use_im_start_end"]:
            im_start_token = self.tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN]
            )[0]
            if input_ids[-1] == im_start_token:
                input_ids = input_ids[:-1]

        n_im_patch = (input_ids == im_patch_token).sum().item()

        if n_im_patch == 0:  # all the images are trimmed
            images = None
        else:
            assert n_im_patch % cur_token_len == 0
            images = images[: n_im_patch // cur_token_len]

        targets = input_ids.clone()

        if self.image_following_text_only:  # keep only text after leading image token
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < targets.shape[-1] and targets[label_idx] != im_patch_token
            ):
                targets[label_idx] = IGNORE_INDEX
                label_idx += 1

            pad_token = self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.pad_token]
            )[0]

            pad_token_idxs = torch.where(targets == pad_token)[0]
            for pad_token_idx in pad_token_idxs:
                token_idx = pad_token_idx + 1
                while (
                    token_idx < targets.shape[-1]
                    and targets[token_idx] != im_patch_token
                ):
                    targets[token_idx] = IGNORE_INDEX
                    token_idx += 1
            # do not train on padding tokens
            targets[targets == pad_token] = IGNORE_INDEX

        # mask image tokens
        # TODO: if we add \n to divide the image later, we need to also mask it here.
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

        return dict(input_ids=input_ids, labels=targets, image=images)


class LazyMMC4DatasetSub(LazyMMC4Dataset):
    def __init__(self, *args, **kwargs):
        kwargs["image_following_text_only"] = True
        super().__init__(*args, **kwargs)


class LazyCoyoDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        multimodal_cfg: dict,
        n_samples_per_idx=4,
    ):
        super().__init__()

        import pickle

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        count_info_list = sorted(
            [f for f in os.listdir(data_path) if f.endswith(".count")]
        )
        n_samples = [
            int(open(os.path.join(data_path, f), "r").read().strip())
            for f in count_info_list
        ]

        print("total COYO samples", sum(n_samples))

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [
            sum(n_samples[i * shared_size : (i + 1) * shared_size]) // n_samples_per_idx
            for i in range(world_size)
        ]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)

        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                shard_data = pickle.load(f)
                random.seed(42)
                if "mmc4" in data_path:
                    random.shuffle(shard_data)  # shuffle for MMC4cap only
                full_data_list.extend(shard_data)

        print("* loaded totally {} samples".format(len(full_data_list)))

        # now pack the samples into groups
        n_groups = len(full_data_list) // n_samples_per_idx
        full_data_list = [
            full_data_list[i : i + n_samples_per_idx]
            for i in range(0, len(full_data_list), n_samples_per_idx)
        ]
        if len(full_data_list[-1]) < n_samples_per_idx:
            full_data_list = full_data_list[:-1]
        assert len(full_data_list) == n_groups
        print("split into {} groups".format(n_groups))

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        info_list = self.data_list[i - self.idx_offset]

        text_list = []
        image_list = []

        for sample in info_list:
            caption_key = "text" if "text" in sample else "caption"
            text_list.append(
                DEFAULT_IMAGE_TOKEN + sample[caption_key] + self.tokenizer.eos_token
            )
            if "image" in sample:
                image_base64 = sample["image"]
                rawbytes = base64.b64decode(image_base64)
            else:
                rawbytes = sample["rawbytes"]
            image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
            image_list.append(image)

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

from functools import lru_cache

@lru_cache(maxsize=16)
def lru_json_load(jpath):
    return json.load(open(jpath, "r"))

class LazyCoyoFull(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        multimodal_cfg: dict,
    ):
        super().__init__()
        
        from llava.train.simple_coyo_dataset import SimpleCoyoDataset
        
        self.dataset = SimpleCoyoDataset(
            data_path=data_path,
        )

        # None: use original caption
        # folder path: use original caption
        self.caption_chocie = None
        self.data_path = data_path
        
        print("total samples", len(self.dataset))  # 10,881,869
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # print("Loading done. Total time: {:.2f} seconds".format(t2 - t1))

        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ADD_TEXT_PROMPT = False

        info = self.dataset[i]
        
        if ".jpg" in info:
            caption, image_path = info[".txt"], info[".jpg"]
        elif ".png" in info:
            caption, image_path = info[".txt"], info[".png"]
        elif ".webp" in info:
            caption, image_path = info[".txt"], info[".webp"]
        elif ".bmp" in info:
            caption, image_path = info[".txt"], info[".bmp"]
        elif ".tiff" in info:
            caption, image_path = info[".txt"], info[".tiff"]
        else:
            print(info.keys())
            print(info)
            raise KeyError
        # except KeyError as e:
        #     print(info.keys())
        #     print(info)
        #     raise e
        
        if self.caption_chocie is not None:
            # load new captions 
            shard = info["__shard__"]
            url = info[".json"]["url"]
            tar_name = osp.relpath(osp.realpath(self.data_path), osp.realpath(self.data_path))
            shard_json_path = osp.join(self.caption_chocie, tar_name + ".json")
            shard_json = lru_json_load(shard_json_path)
            try:
                caption = shard_json[url]["output"]
            except KeyError:
                print(f"{url} not in caption. fallback to original caption temporarially")
                
            
        if ADD_TEXT_PROMPT:
            from llava.data.template import CAPTION_TEMPLATE

            rand_prompt = random.choice(CAPTION_TEMPLATE)
            rand_prompt = "<image>\n" + rand_prompt
        else:
            rand_prompt = "<image>\n"
        sources = [
            {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": rand_prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        ]

        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        if "image" in sources[0]:
            image = LazySupervisedDataset._process_image(
                sources[0]["image"], self.multimodal_cfg
            )
            image = torch.unsqueeze(image, dim=0)

            # now random pick some context samples for training
            if self.multimodal_cfg["num_shots"] > 0:
                raise NotImplementedError
            
            # the same size for all images, so we concat
            cur_token_len = (image.shape[-2] // self.multimodal_cfg["patch_size"]) * (
                image.shape[-1] // self.multimodal_cfg["patch_size"]
            )
            cur_token_len += self.multimodal_cfg["n_extra_patch"]
            sources = replace_image_patch_tokens(
                [e["conversations"] for e in sources], self.multimodal_cfg
            )
        else:
            raise NotImplementedError

        if not ADD_TEXT_PROMPT:
            assert len(sources) == 1
            # tokenize conversations
            image_tokens = tokenizer_image_token(
                sources[0][0]["value"],
                self.tokenizer,
                n_image_tokens=cur_token_len,
                return_tensors="pt",
            ).view(1, -1)
            text_tokens = self.tokenizer(
                [sources[0][1]["value"] + "</s>"],
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids
            input_ids = torch.cat([image_tokens, text_tokens[:, 1:]], dim=-1)
            targets = input_ids.clone()

            targets[:, : image_tokens.shape[-1]] = IGNORE_INDEX
            data_dict = dict(input_ids=input_ids, labels=targets)

        else:
            data_dict = preprocess(
                sources, self.tokenizer, n_image_tokens=cur_token_len
            )

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        if image is not None:
            data_dict["image"] = image
        else:
            raise NotImplementedError

        return data_dict



class LazyCoyoFullRecaptioned(LazyCoyoFull):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.caption_chocie = "/home/ligengz/workspace/VILA/captioner"
        
        print(f"Loading recaptioned texts from VILA {self.caption_chocie}")
        


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    concat_prob: 0.0

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # TODO(ligeng): what is this check for?
        if self.concat_prob > 0.0 and random.random() < self.concat_prob:
            raise NotImplementedError
        # how to support mixing text only & text image???
        # NOTE(ligeng): temporally disable the check for webcoyo dataset
        assert all(["image" in instance for instance in instances])
        # print(instances)
        input_ids, labels, images = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "image")
        )
        # exit(0)
        new_input_ids, new_labels, new_images = [], [], []
        for input_id, label, image in zip(input_ids, labels, images):
            if input_id.dim() > 1:
                new_input_ids.extend(
                    [f.squeeze(0) for f in input_id.chunk(input_id.shape[0], dim=0)]
                )
                new_labels.extend(
                    [f.squeeze(0) for f in label.chunk(input_id.shape[0], dim=0)]
                )
                new_images.extend(
                    image.chunk(input_id.shape[0], dim=0)
                )
            else:
                new_input_ids.append(input_id)
                new_labels.append(label)
                new_images.append(image)
        input_ids, labels, images = new_input_ids, new_labels, new_images

        assert len(input_ids) == len(images)
        # Sort the input_ids by length in descending order
        combined = sorted(zip(input_ids, labels, images), key=lambda x: len(x[0]), reverse=True)
        sorted_ids, sorted_labels, sorted_images = zip(*combined)

        max_seq_length = len(sorted_ids[0])

        batches = []
        label_batches = []
        seqlens_in_batch = []
        position_ids = []
        batch_images = []

        while sorted_ids:
            current_batch = torch.tensor([], dtype=torch.int32)
            current_label_batch = torch.tensor([], dtype=torch.int32)
            current_position_ids = torch.tensor([], dtype=torch.int32)
            i = 0
            while i < len(sorted_ids):
                if len(current_batch) + len(sorted_ids[i]) <= max_seq_length:
                    seqlens_in_batch.append(sorted_ids[i].ne(self.tokenizer.pad_token_id).sum())
                    current_position_ids = torch.cat((current_position_ids, torch.arange(start=0, end=len(sorted_ids[i]))), dim=0)
                    current_batch = torch.cat((current_batch, sorted_ids[i]), dim=0)
                    current_label_batch = torch.cat((current_label_batch, sorted_labels[i]), dim=0)
                    sorted_ids = sorted_ids[:i] + sorted_ids[i+1:]
                    sorted_labels = sorted_labels[:i] + sorted_labels[i+1:]

                    if sorted_images[i] is not None:
                        batch_images.append(sorted_images[i])
                    sorted_images = sorted_images[:i] + sorted_images[i+1:]
                else:
                    i += 1

            batches.append(current_batch)
            label_batches.append(current_label_batch)
            position_ids.append(current_position_ids)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batches, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            label_batches, batch_first=True, padding_value=IGNORE_INDEX
        )
        seqlens_in_batch = torch.stack(seqlens_in_batch, axis=0)
        position_ids = torch.nn.utils.rnn.pad_sequence(
            position_ids, batch_first=True, padding_value=IGNORE_INDEX
        )

        # assert len(batch_images) == torch.sum(input_ids==32000)/576
        if batch_images:
            flat_batch_images = torch.concat(batch_images, dim=0)
        else:
            flat_batch_images = None

        assert seqlens_in_batch.sum() == input_ids.ne(self.tokenizer.pad_token_id).flatten().sum()

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # notice that we inject attention mask here
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            seqlens_in_batch=seqlens_in_batch,
            images = flat_batch_images,
            position_ids=position_ids
        )
        # rprint("input_ids: ", batch["input_ids"].shape, "labels: ", batch["labels"].shape)
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    patch_size,
    image_size,
    n_extra_patch=0,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    extra_info = []
    all_datasets = []
    # print(datasets_mixture.DATASETS_MIXTURES)
    mixture = datasets_mixture.DATASETS_MIXTURES[data_args.datasets_mixture_name]
    for dataset in mixture:
        dataset_type = dataset.dataset_type
        if dataset_type == "torch":
            dataset_cls = (
                LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
            )
        elif dataset_type == "wds":
            dataset_cls = LazyWDSDataset
        elif dataset_type == "vflan":
            dataset_cls = LazyVFlanDataset
        elif dataset_type == "mmc4":
            dataset_cls = LazyMMC4Dataset
        elif dataset_type == "mmc4sub":
            dataset_cls = LazyMMC4DatasetSub
        elif dataset_type == "coyo":
            dataset_cls = LazyCoyoDataset
        elif dataset_type == "coyowebds":
            print("dataset.py: Loading LazyCoyoFull class")
            dataset_cls = LazyCoyoFull
        elif dataset_type == "coyowebds_recap":
            print("dataset.py: Loading LazyCoyoFull class with captioned results")
            dataset_cls = LazyCoyoFullRecaptioned
        else:
            raise NotImplementedError


        train_dataset = dataset_cls(
            tokenizer=tokenizer,
            data_path=dataset.data_path,
            multimodal_cfg=dict(
                num_shots=data_args.num_shots,
                is_multimodal=data_args.is_multimodal,
                sep_image_conv_front=data_args.sep_image_conv_front,
                image_token_len=data_args.image_token_len,
                image_folder=dataset.image_path,
                image_aspect_ratio=data_args.image_aspect_ratio,
                use_im_start_end=getattr(data_args, "mm_use_im_start_end", False),
                image_processor=getattr(data_args, "image_processor", None),
                patch_size=patch_size,
                image_size=image_size,
                n_extra_patch=n_extra_patch,
            ),
        )
        all_datasets.append(train_dataset)
        extra_info.append(len(train_dataset))
    
    if len(all_datasets) > 1:
        print("Concatnating datasets")
    all_datasets = ConcatDataset(all_datasets)
    # disable nvgpt4 utils
    # from nvgpt4.data import BlendDataset, WorkerConfig
    # rank = int(os.environ.get("RANK", 0))
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # all_datasets = BlendDataset(
    #     *[(dst, 1) for dst in all_datasets],
    #     worker_config=WorkerConfig(
    #         rank=rank,
    #         world_size=world_size,
    #         num_workers=1,
    #     )
    # )

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, concat_prob=0.0
    )  # whether to concat
    return (
        dict(
            train_dataset=all_datasets, eval_dataset=None, data_collator=data_collator
        ),
        extra_info,
    )
