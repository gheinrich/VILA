import argparse
import base64
import io
import json
import os
import pickle

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
)
from transformers.models.siglip import (
    SiglipImageProcessor,
)

from llava import LlavaLlamaForCausalLM
from llava.train.dataset import LazyMMC4Dataset
from llava.utils import disable_torch_init

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


class LazyMMC4Dataset2(LazyMMC4Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        multimodal_cfg: dict,
        image_following_text_only=False,
    ):
        Dataset.__init__(self)

        import pickle

        shard_names = [f for f in os.listdir(data_path) if f.endswith(".pkl")]
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

        self.n_samples = len(self.data_list)
        self.idx_offset = 0

        self.text_only = False


def chamfer_distance_cosine(x, y):
    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)

    dists1 = x_norm @ y_norm.t()
    dists2 = y_norm @ x_norm.t()

    max_dists1, _ = torch.max(dists1, dim=1)  # Min distances from set1 to set2
    max_dists2, _ = torch.max(dists2, dim=1)  # Min distances from set2 to set1

    chamfer_dist = max_dists1.mean() + max_dists2.mean()
    return (chamfer_dist / 2).item()


def chamfer_distance(set1, set2):
    """
    Compute the Chamfer distance between two point sets.

    Parameters:
    - set1, set2: Input point sets, Shape: [batch_size, num_points, dim]

    Returns:
    - Chamfer distance
    """
    dists1 = torch.cdist(
        set1, set2, p=2
    )  # Pairwise distances [batch_size, num_points_set1, num_points_set2]
    dists2 = torch.cdist(set2, set1, p=2)

    min_dists1, _ = torch.min(dists1, dim=1)  # Min distances from set1 to set2
    min_dists2, _ = torch.min(dists2, dim=1)  # Min distances from set2 to set1

    chamfer_dist = min_dists1.mean() + min_dists2.mean()

    return chamfer_dist.item()


def get_attention_map(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda()

    if "siglip" in args.model_name:
        image_processor = SiglipImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16
        )
    else:
        image_processor = CLIPImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16
        )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    vision_tower = model.get_model().vision_tower[0]
    vision_tower.to(device="cuda", dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )
    if "qwen" in model_name.lower():  # TODO: a more elegant way
        vision_config.patch_size = 28
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    if not hasattr(model.config, "mm_projector_type"):
        model.config.mm_projector_type = "linear"

    if (
        "downsample" in model.config.mm_projector_type
        or "ds" in model.config.mm_projector_type
    ):
        image_token_len = image_token_len // 4

    if "p32" in args.model_name:  # extra leading patches
        n_extra_patch = 32
    elif "se" in model.config.mm_projector_type:
        n_extra_patch = 2
    else:
        n_extra_patch = 0

    print(image_token_len)

    dataset = LazyMMC4Dataset2(
        data_path="/home/jil/datasets/mmc4-core/pkl-val-sub",
        tokenizer=tokenizer,
        multimodal_cfg=dict(
            num_shots=0,
            is_multimodal=True,
            sep_image_conv_front=False,
            image_token_len=image_token_len,
            image_folder=None,
            image_aspect_ratio="square",
            use_im_start_end=False,
            image_processor=image_processor,
            patch_size=vision_config.patch_size,
            n_extra_patch=n_extra_patch,
        ),
        image_following_text_only=True,
    )

    print(len(dataset))

    n_sample = 50
    # n_sample = 100
    for i_data, data in enumerate(dataset):
        if i_data >= n_sample:
            break
        # print(data["input_ids"].shape)
        # print(data["image"].shape)
        with torch.inference_mode():
            output = model(
                data["input_ids"].unsqueeze(0).cuda(),
                images=data["image"].half().cuda(),
                output_hidden_states=True,
                output_attentions=True,
            )

        image_idx = data["input_ids"] == 32000
        text_idx = (data["input_ids"] != 32000) & (
            data["input_ids"] != tokenizer.pad_token_id
        )

        print(image_idx)
        image_idx = image_idx.cuda()
        text_idx = text_idx.cuda()
        for i in range(len(output.attentions)):
            print(output.attentions[i].shape)
        attentions = [attn.mean(0).mean(0) for attn in output.attentions]

        torch.save((image_idx, attentions), "sample_attn.pt")

        exit()
    # now calculate the kl-divergence between visual and text


def measure_vis_emb_sim(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda()

    if "siglip" in args.model_name:
        image_processor = SiglipImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16
        )
    else:
        image_processor = CLIPImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16
        )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    vision_tower = model.get_model().vision_tower[0]
    vision_tower.to(device="cuda", dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        (
            vision_config.im_start_token,
            vision_config.im_end_token,
        ) = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
        )
    if "qwen" in model_name.lower():  # TODO: a more elegant way
        vision_config.patch_size = 28
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    if not hasattr(model.config, "mm_projector_type"):
        model.config.mm_projector_type = "linear"

    if (
        "downsample" in model.config.mm_projector_type
        or "ds" in model.config.mm_projector_type
    ):
        image_token_len = image_token_len // 4

    if "p32" in args.model_name:  # extra leading patches
        n_extra_patch = 32
    elif "se" in model.config.mm_projector_type:
        n_extra_patch = 2
    else:
        n_extra_patch = 0

    print(image_token_len)

    dataset = LazyMMC4Dataset2(
        data_path="/home/jil/datasets/mmc4-core/pkl-val-sub",
        tokenizer=tokenizer,
        multimodal_cfg=dict(
            num_shots=0,
            is_multimodal=True,
            sep_image_conv_front=False,
            image_token_len=image_token_len,
            image_folder=None,
            image_aspect_ratio="square",
            use_im_start_end=False,
            image_processor=image_processor,
            patch_size=vision_config.patch_size,
            n_extra_patch=n_extra_patch,
        ),
        image_following_text_only=True,
    )

    print(len(dataset))

    n_sample = 50
    # n_sample = 100
    for i_data, data in enumerate(dataset):
        if i_data >= n_sample:
            break

        vision_tower = model.get_vision_tower()
        images = data["image"].half().cuda()
        vis_out = vision_tower(images, output_hidden_states=True)
        # for out in vis_out.hidden_states:
        #     print(out.shape)
        vis_hidden = [out[1, 1:] for out in vis_out.hidden_states]
        for i, hidden in enumerate(vis_hidden):
            hidden = hidden / hidden.norm(dim=1, keepdim=True)
            cos = hidden @ hidden.T
            print(i, torch.mean(cos).item())

            # print(hidden.shape)

        exit()

        # print(data["input_ids"].shape)
        # print(data["image"].shape)
        with torch.inference_mode():
            output = model(
                data["input_ids"].unsqueeze(0).cuda(),
                images=data["image"].half().cuda(),
                output_hidden_states=True,
                output_attentions=True,
            )

        image_idx = data["input_ids"] == 32000
        text_idx = (data["input_ids"] != 32000) & (
            data["input_ids"] != tokenizer.pad_token_id
        )

        print(image_idx)
        image_idx = image_idx.cuda()
        text_idx = text_idx.cuda()
        for i in range(len(output.attentions)):
            print(output.attentions[i].shape)
        attentions = [attn.mean(0).mean(0) for attn in output.attentions]

        torch.save((image_idx, attentions), "sample_attn.pt")

        exit()
    # now calculate the kl-divergence between visual and text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")

    args = parser.parse_args()

    # get_attention_map(args)
    measure_vis_emb_sim(args)


# if __name__ == "__main__":
#     dataset = MMBenchDataset("/home/jil/datasets/mmbench/mmbench_dev_20230712.tsv")
#     print(dataset[100])
