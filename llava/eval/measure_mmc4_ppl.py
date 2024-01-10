import argparse
import base64
import io
import json
import os
import pickle

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
)

from llava import LlavaLlamaForCausalLM
from llava.model.visual_attn_scale import new_attention_forward
from llava.train.dataset import LazyMMC4Dataset
from llava.utils import disable_torch_init

transformers.models.llama.modeling_llama.LlamaAttention.forward = new_attention_forward

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
        text_only=False,
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
        self.text_only = text_only

        self.n_samples = len(self.data_list)
        self.idx_offset = 0


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda()

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
        image_following_text_only=False,
        text_only=args.text_only,
    )

    print(len(dataset))

    ## llama2-7b-finetune-mmc4sub-linear-e1-nose-run2
    # 200 image_following_text_only=True  7.4207470703125
    # 200   9.773740234375
    # 400   9.3576416015625
    # 400 no-image 9.99474853515625
    ## llama2-7b-finetune-mmc4sub+coyo-linear-e1-nose-accum-run2
    # 400 8.5366650390625

    ppl_list = []

    n_sample = 1000
    # n_sample = 100
    for i_data, data in tqdm(enumerate(dataset), total=n_sample):
        if i_data >= n_sample:
            break
        # print(data["input_ids"].shape)
        # print(data["image"].shape)
        with torch.inference_mode():
            output = model(
                data["input_ids"].unsqueeze(0).cuda(),
                images=data["image"].half().cuda(),
                labels=data["labels"].unsqueeze(0).cuda(),
                output_hidden_states=True,
            )

            loss = output.loss
            ppl = torch.exp(loss).item()
            ppl_list.append(ppl)
            print(ppl)

    print(sum(ppl_list) / len(ppl_list))
    if args.out_path is not None:
        with open(args.out_path, "w") as f:
            json.dump(ppl_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--out_path", type=str, default=None)

    args = parser.parse_args()

    eval_model(args)


# if __name__ == "__main__":
#     dataset = MMBenchDataset("/home/jil/datasets/mmbench/mmbench_dev_20230712.tsv")
#     print(dataset[100])
