import argparse
import json
import os
import os.path as osp
from glob import glob
from itertools import chain
from pprint import pprint
from typing import Any, Dict

import numpy as np
import torch
from datasets import load_dataset
from hydra.utils import instantiate
from PIL import Image
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, ProcessorMixin

from llava.data.builder import DATASETS, parse_mixture
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger
from llava.utils.media import extract_media


class VQAv2Dataset(Dataset):
    def __init__(
        self,
        data,
    ):
        self.data = load_dataset("lmms-lab/VQAv2", split="validation")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "image": self.data["image"][idx],
            "question": data["question"][idx] + "###" + data["multiple_choice_answer"][idx],
            "answer": data["answers"][idx],
        }


def get_vqav2():
    data = load_dataset("lmms-lab/VQAv2", split="validation")
    print("prefetching images vqav2")
    img_source_list = []
    for idx, img in enumerate(tqdm(data["image"])):
        img_source_list.append(
            {
                "image": img,
                "question": data["question"][idx] + "###" + data["multiple_choice_answer"][idx],
                "answer": data["answers"][idx],
            }
        )
    print("prefetching images vqav2 finished")
    for data in img_source_list:
        yield data


def get_scienceqa():
    # NOTE(ligeng): some images are None
    data = load_dataset("lmms-lab/ScienceQA", "ScienceQA-FULL", split="test")
    print("prefetching images")
    img_source_list = []
    for idx, img in enumerate(tqdm(data["image"])):
        yield {
            "image": img,
            "question": data["question"][idx] + "###" + str(data["choices"][idx]),
            "answer": data["answer"][idx],
        }


def get_ai2d():
    data = load_dataset("lmms-lab/ai2d", split="test")
    print("prefetching images")
    img_source_list = []
    for idx, img in enumerate(tqdm(data["image"])):
        yield {
            "image": img,
            "question": data["question"][idx] + "###" + str(data["options"][idx]),
            "answer": data["answer"][idx],
        }


def get_mmmu_val():
    data = load_dataset("lmms-lab/MMMU", split="validation")
    print("prefetching images")
    img_source_list = []
    for idx, img in enumerate(tqdm(data["image_1"])):
        yield {
            "image": img,
            "question": data["question"][idx] + "###" + data["options"][idx],
            "answer": data["answer"][idx],
        }


def get_docvqa():
    data = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    for idx, img in enumerate(data["image"]):
        yield {
            "image": img,
            "question": data["question"][idx],
            "answer": data["answers"][idx],
        }


def get_chartqa():
    data = load_dataset("lmms-lab/ChartQA", split="test")
    for idx, img in enumerate(data["image"]):
        yield {
            "image": img,
            "question": data["question"][idx],
            "answer": data["answer"][idx],
        }


def get_mme():
    data = load_dataset("lmms-lab/MME", split="test")
    for idx, img in enumerate(data["image"]):
        yield {
            "image": img,
            "question": data["question"][idx],
            "answer": data["answer"][idx],
        }


def get_gqa():
    data = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    for img in data["image"]:
        yield {
            "image": img,
            "question": None,
            "answer": None,
        }


def get_ocrbench():
    data = load_dataset("echo840/OCRBench", split="test")
    for idx, img in enumerate(data["image"]):
        yield {
            "image": img,
            "question": data["question"][idx],
            "answer": data["answer"][idx],
        }


def get_textvqa():
    data = load_dataset("lmms-lab/TextVQA", split="validation")
    for idx, img in enumerate(data["image"]):
        yield {
            "image": img,
            "question": data["question"][idx],
            "answer": data["answers"][idx],
        }


task2fn = {
    "mmmu_val": get_mmmu_val,
    # "ai2d": get_ai2d,
    # "gqa": get_gqa,
    # "docvqa": get_docvqa,
    # "chartqa": get_chartqa,
    # "mme": get_mme,
    # "ocrbench": get_ocrbench,
    # "textvqa": get_textvqa,
    # "scienceqa": get_scienceqa, # buggy
    # "vqav2": get_vqav2,
}


class FeatureExtractor:
    def __init__(self, model_name_or_path: str = "google/siglip-so400m-patch14-384"):
        self.model = AutoModel.from_pretrained(model_name_or_path).cuda()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def __call__(self, image: Image.Image, text: str = None) -> np.ndarray:
        # TODO: support batched inference
        if isinstance(image, str):
            assert os.path.exists(image), f"Image file {image} does not exist"
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt").to("cuda")
        with torch.inference_mode(), torch.cuda.amp.autocast():
            features = self.model.get_image_features(**inputs)
        results = {
            "img_feat": features,
        }

        if text is not None:
            text_emd = self.tokenizer(text=text, return_tensors="pt", truncation=True)
            text_input_ids = text_emd["input_ids"][:, :64]
            text_pos_ids = torch.arange(text_input_ids.shape[1]).repeat(text_input_ids.shape[0], 1)
            text_features = self.model.get_text_features(
                input_ids=text_input_ids.to("cuda"), position_ids=text_pos_ids.to("cuda")
            )
            results["text_feat"] = text_features

        return results


def main(
    topk_nums=300, topk_threshold=0.75, mmdb_dir="data_curation_dev/mmdb", output_dir="data_curation_dev/val_siglip"
):
    features, text_features, metainfos = [], [], []
    # merge all text and image embeddings
    dataset_len_info = {
        "name": [],
        "count": [],
    }
    for fpath in glob(os.path.join(mmdb_dir, "*.jsonl")):
        features.append(io.load(fpath.replace(".jsonl", ".pt"), map_location="cuda"))
        text_features.append(io.load(fpath.replace(".jsonl", "_text.pt"), map_location="cuda"))
        jsonl_info = io.load(fpath)
        metainfos.extend(jsonl_info)
        dataset_name = osp.basename(fpath)
        dataset_len_info["name"].append(dataset_name)
        dataset_len_info["count"].append(len(jsonl_info))

    # print(len(features), len(text_features))
    features = torch.cat(features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    # all_features = torch.cat((features, text_features), dim=-1)

    logger.info(f"Loaded {len(features)} SFT image & text features from '{mmdb_dir}' with unique paths.")
    # print(features.shape, text_features.shape, all_features.shape)

    ft = FeatureExtractor("google/siglip-so400m-patch14-384")
    val2sft_info_all = {}

    @torch.no_grad()
    def topk_filter(f, all_f, topk=100, topk_thres=0.75):
        _scores = cosine_similarity(f, all_f)
        scores, indices = _scores.topk(topk)
        mask = scores > topk_thres
        scores = scores[mask]
        indices = indices[mask]
        return _scores, scores, indices

    os.makedirs(output_dir, exist_ok=True)
    for task, fn in task2fn.items():
        val2sft_info = {}
        if osp.exists(f"{output_dir}/{task}.json") and osp.exists(f"{output_dir}/{task}.pth"):
            print("`skipping` ", task)
            continue
        print("retrieving images for ", task)
        sum_scores = None
        for didx, data in enumerate(fn()):
            img = data["image"]
            img_path = f"{task}/{didx}.png"
            all_feats = ft(img, text=str(data["question"]))
            feats = all_feats["img_feat"]
            txt_feats = all_feats["text_feat"]

            all_scores, scores, indices = topk_filter(feats, features, topk=topk_nums, topk_thres=topk_threshold)
            if sum_scores is None:
                sum_scores = all_scores
            else:
                sum_scores += all_scores
            print("Extracting features for", img_path, scores.shape, all_scores.shape)

            for index, score in zip(indices, scores):
                metainfo = metainfos[index.item()]
                # sft -> val mapping
                mpath = metainfo["path"]
                response = {
                    "uid": metainfo["uid"],
                    "path": metainfo["path"],
                    "score": score.item(),
                }
                # sft -> val mapping
                if img_path not in val2sft_info:
                    val2sft_info[img_path] = {
                        "question": str(data["question"]),
                        "answer": str(data["answer"]),
                        "responses": [],
                    }

                val2sft_info[img_path]["responses"].append(response)

        print("Finish retrieving images for ", task, " now saving")
        json.dump(val2sft_info, open(f"{output_dir}/{task}.json", "w"), indent=2)
        torch.save(sum_scores, f"{output_dir}/{task}.pth")


if __name__ == "__main__":
    main()
