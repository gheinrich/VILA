import os

import torch
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

from llava.conversation import SeparatorStyle, conv_templates
from llava.model import LlavaLlamaForCausalLM
from llava.train.token_config import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)
from llava.utils import disable_torch_init


class RGBCheck:
    def __call__(self, pic):
        assert len(pic.shape) == 3
        if pic.shape[0] == 1:
            pic = pic.repeat(3, 1, 1)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToRGB:
    def __call__(self, pic):
        return pic.convert("RGB")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def build_model(model_name, conv_version):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    # assert mm_use_im_start_end
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

    if "p32" in model.config.mm_projector_type:
        n_extra_patch = 32
    elif "se71" in model.config.mm_projector_type:
        n_extra_patch = 8
    elif "se" in model.config.mm_projector_type:
        n_extra_patch = 2
    elif "repeat" in model.config.mm_projector_type:
        n_extra_patch = 256
    else:
        n_extra_patch = 0

    if "sam" in str(type(model.get_vision_tower())).lower():
        patch_size = 16
    elif (
        "visiontransformer" in str(type(model.get_vision_tower())).lower()
        and model.config.mm_projector_type != "dsresampler"
        and "eva" not in str(type(model.get_vision_tower())).lower()
    ):
        patch_size = 28  # qwen
    else:  # clip
        patch_size = 14
    patch_size = patch_size * 2 ** model.config.mm_projector_type.count("ds")

    image_token_len = (vision_config.image_size // patch_size) ** 2 + n_extra_patch

    if mm_use_im_start_end:
        image_tokens = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
        )
    else:
        image_tokens = DEFAULT_IMAGE_TOKEN + "\n"

    if conv_version in ["v1", "vicuna_v1_1"]:
        tokenizer.pad_token = tokenizer.unk_token
        conv = conv_templates["vicuna_v1_1"].copy()
    elif conv_version in ["caption", "vqa", "vicuna_v1_1_nosys"]:
        tokenizer.pad_token = tokenizer.unk_token
        conv = conv_templates[conv_version].copy()
    elif conv_version == "v0":
        print(
            "Legacy v0 model eval... should only be applied for the official models..."
        )
        assert "v0" in model_name
        from llava.train.token_config import (
            DEFAULT_BOS_TOKEN,
            DEFAULT_EOS_TOKEN,
            DEFAULT_PAD_TOKEN,
            DEFAULT_UNK_TOKEN,
        )

        if tokenizer.pad_token is None:
            from llava.train.train import smart_tokenizer_and_embedding_resize

            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
        conv = conv_templates["default"].copy()
    else:
        raise NotImplementedError

    model.eval()

    # print("debug" * 20, image_token_len, vision_config.image_size)
    return model, tokenizer, image_processor, conv, image_tokens, image_token_len


def preprocess_image(image, processor, use_padding=False, pad_color="mean"):
    from PIL import Image

    if use_padding:

        def expand2square(pil_img, background_color):
            pil_img = pil_img.convert("RGB")
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

        if pad_color == "mean":
            pad_color = tuple(int(x * 255) for x in processor.image_mean)
        elif pad_color == "white":
            pad_color = (255, 255, 255)
        else:
            raise NotImplementedError
        image = expand2square(image, pad_color)
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"]
    else:
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"]

    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    return image
