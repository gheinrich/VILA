from filelock import Timeout, FileLock
import shutil
import torch
import pprint
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import AutoTokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

from llava.train import datasets_mixture
from llava.train.dataset_coyo_test import LazyCoyoDataset
from llava.train.arguments import DataArguments, ModelArguments, TrainingArguments

import argparse
import glob, os.path as osp, os
from io import BytesIO
import json
import requests

import torch
import transformers
from transformers import logging
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    StoppingCriteria,
    AutoModel,
)

from llava.conversation import SeparatorStyle, conv_templates
from llava.eval.utils import preprocess_image
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria
from llava.model.visual_attn_scale import new_attention_forward
from llava.utils import disable_torch_init

transformers.models.llama.modeling_llama.LlamaAttention.forward = new_attention_forward

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    '''
    [4879083473105, 'https://cdn.billiger.com/dynimg/iBpF8x19A1EeE6JWhZ4CUgA2-pEoXYO2FO0obcY2xnQ1YO06rOi28g98iBnbjTFUopXq5ZfhHBQqF1VM8lIcu26sKkZG1CqYItu6E_XkUrRJATRZBfIhttOPYy5HiC-CEfUD0VilOp6Da-X9DPpbmdzQ7_-pwCreVTNv4QUAJ7hPqVE2WFUAuxagDi9LZMVqA/2061311384_large.png', 'AVM FRITZ!Repeater 1200 WLAN Mesh (866Mbit/s, 400Mbit/s), WLAN Repeater']
    '''
    h, w = image.size
    if h < 10 and w < 10:
        image = image.resize((30, 30))
    return image


def execute_llava(
    model,
    model_name,
    image_processor,
    tokenizer,
    vision_config,
    image_files: str = None,
    prompt_query: str = None,
    image_rawbytes=None,
):
    mm_use_im_start_end = vision_config.use_im_start_end
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
    if (
        "downsample" in model.config.mm_projector_type
        or "ds" in model.config.mm_projector_type
    ):
        image_token_len = image_token_len // 4

    if "p32" in args.model_name:  # extra leading patches
        image_token_len += 32
    elif "se" in model.config.mm_projector_type:
        image_token_len += 2

    # read images first
    if image_rawbytes is None:
        image_file_list = image_files.split("###")
        image_list = [load_image(image_file) for image_file in image_file_list]
    else:
        image_list = []
        _image_list = [
            Image.open(BytesIO(rawbytes)).convert("RGB") for rawbytes in image_rawbytes
        ]
        for image in _image_list:
            h, w = image.size
            if h < 10 and w < 10:
                image = image.resize((30, 30))
            image_list.append(image)

    image_tensor = torch.cat(
        [
            preprocess_image(img, image_processor, use_padding=args.pad)
            for img in image_list
        ],
        dim=0,
    )

    # print("Analyzing input of tensor images' shape:", image_tensor.shape)

    query = prompt_query
    if query.endswith(".txt"):
        with open(query) as f:
            lines = f.readlines()
        query = "".join([l.strip() for l in lines])

    if mm_use_im_start_end:
        image_tokens = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            + DEFAULT_IM_END_TOKEN
        )
    else:
        image_tokens = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

    if not "<image>" in query:
        assert "###" not in query  # single query
        query = image_tokens + "\n" + query  # add <image>
        query_list = [query]
    else:
        query_list = query.split("###")
        assert len(query_list) % 2 == 1  # the last one is from human

        new_query_list = []
        for i, query in enumerate(query_list):
            if "<image>" in query:
                assert i % 2 == 0  # only from human
                query = query.replace("<image>", image_tokens)
            new_query_list.append(query)
        query_list = new_query_list

    if "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt_multimodal"
    else:
        conv_mode = "multimodal"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print("Now using conversation mode {}".format(args.conv_mode))
        # print(
        #    "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
        #        conv_mode, args.conv_mode, args.conv_mode
        #    )
        # )
    else:
        args.conv_mode = conv_mode

    if args.manual_prompt is not None:
        prompt = args.manual_prompt.replace("<image>", image_tokens)
    else:
        conv = conv_templates[args.conv_mode].copy()

        for i, query in enumerate(query_list):
            conv.append_message(conv.roles[i % 2], query)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

    # print(prompt)
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # print("Analyzing input_ids of shape:", input_ids.shape)

    if args.manual_prompt is not None:
        stop_str = "</s>"
    else:
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            max_new_tokens=512,
            # top_p=0.7,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    return outputs


def safely_merge_info(out_fpath, info):
    os.makedirs(osp.dirname(out_fpath), exist_ok=True)
    with FileLock(out_fpath.replace(".json", ".lock")):
        if osp.exists(out_fpath):
            new_info = json.load(
                open(out_fpath, "r+"),
            )
            info.update(new_info)
        json.dump(info, open(out_fpath + ".meta", "w+"), indent=2)
        shutil.move(out_fpath + ".meta", out_fpath)


def eval_model(args, idx, total):
    print("launch distributed mode")
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print("INFO:", local_rank, rank, world_size)
    torch.cuda.set_device(local_rank)
    
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, trust_remote_code=True
    )

    # Output
    logging.set_verbosity_error()

    if "mpt" in model_name.lower():
        model = LlavaMPTForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_cache=True,
        ).cuda()
    # elif "share" in model_name.lower():
    #     print("loading sharecaptioner")
    #     model = AutoModel.from_pretrained(
    #         model_name,
    #         low_cpu_mem_usage=True,
    #         torch_dtype=torch.float16,
    #         use_cache=True,
    #         trust_remote_code=True,
    #     ).cuda()
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_name,
            # low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_cache=True,
            device_map="cuda",
        )  # .cuda()

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
    if vision_tower.device.type == "meta":
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
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

    dataset = args.dataset
    dataset2fpath = {
        "coco": "~/datasets/ShareGPT4V/data/coco/train2017",
        "sam": "~/datasets/ShareGPT4V/data/sam/images",
    }

    # tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14-336", torch_dtype=torch.float16
    )
    data_args = DataArguments(
        datasets_mixture_name="coyo_25m_mmc4core_test",
        is_multimodal=True,
        lazy_preprocess=True,
    )

    from llava.train.simple_coyo_dataset_vila_ji import SimpleCoyoDataset

    data_path="/home/jasonlu/datasets/coyo-700m/pkl02-split"
    # data_path = "/home/jasonlu/vlm_datasets/debug/coyo-700m/pkl02-split"
    train_dataset = SimpleCoyoDataset(data_path=data_path)
    # sampler = DistributedSampler(train_dataset)
    # dloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, sampler=sampler, num_workers=8)
    # sampler.set_epoch(0)
    
    chunk_size = len(train_dataset) // total
    begin_idx = chunk_size * idx
    stop_idx = chunk_size * (idx + 1)
    subdataset = torch.utils.data.Subset(train_dataset, range(begin_idx, stop_idx))
    sampler = DistributedSampler(subdataset)
    dloader = torch.utils.data.DataLoader(subdataset, shuffle=False, num_workers=12, sampler=sampler)

    print(f"RANKING: {idx} / {total} | CHUNK SIZE: {chunk_size}, {begin_idx} to {stop_idx} | {len(train_dataset)}")

    out_fpath = (
        f"captioner/{dataset}-{osp.basename(model_name)}/{idx}-of-{total}.json"
    )
    out_folder = osp.dirname(out_fpath)
    os.makedirs(out_folder, exist_ok=True)

    info_json = {}
 
    if osp.exists(out_fpath):
        new_info = json.load(open(out_fpath, "r"))
        info_json.update(new_info)
        print(f"loaded {len(new_info.keys())} from {out_fpath}")

   
    for idx, data in enumerate(dloader):
        uuid = data["url"][0]
        orig_text = data["text"][0]
        image_rawbytes = data["image"][0]

        if uuid in info_json:
            continue

        query = "<image> Can you briefly explain the content in the image?"
        output = execute_llava(
            model=model,
            model_name=args.model_name,
            image_processor=image_processor,
            tokenizer=tokenizer,
            vision_config=vision_config,
            prompt_query=query,
            image_rawbytes=[
                image_rawbytes,
            ],
        )

        info_json[uuid] = {
            "query": query,
            "orig_text": orig_text,
            "output": output,
        }

        print(
            "%" * 10
            + " " * 5
            + f"VILA Response [{idx} / {len(dloader)} / {len(train_dataset)}] [split: {idx} / {total}] [rank: {rank} / {world_size}] "
            + " " * 5
            + "%" * 10
        )
        print(uuid, info_json[uuid])

        if idx % 200 == 0:
            safely_merge_info(out_fpath, info_json)
            # safely_merge_info(full_fpath, info_json)
    safely_merge_info(out_fpath, info_json)
    # safely_merge_info(full_fpath, info_json)


if __name__ == "__main__":
    import torch.distributed as dist
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--pad", action="store_true")
    parser.add_argument("--manual_prompt", type=str, default=None)

    # replaced by rank and world size
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--total", type=int, default=-1)

    args = parser.parse_args()
    
    print(args.idx, args.total)
    if args.idx >= 0 and args.total >= 0:
        print("launch individually")
        rank = args.idx
        world_size = args.total
        eval_model(args, rank, world_size)
    else:
        raise NotImplementedError
        print("launch distributed mode")
        dist.init_process_group(backend="nccl")
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        eval_model(args, rank, world_size)
        
    
    
