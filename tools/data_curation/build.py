import argparse
import os
from functools import lru_cache, reduce
from itertools import chain
from pprint import pprint
from typing import Any, Dict

import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, ProcessorMixin

from llava.data.builder import DATASETS, build_dataset, parse_mixture
from llava.data.datasets_mixture import DATASETS_LEGACY, register_datasets_mixtures
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger
from llava.utils.media import extract_media


def add_margin(pil_img, top=0, right=0, bottom=0, left=0, color=(255, 255, 255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


class ImageDataset(Dataset):
    def __init__(self, name: str, processor: ProcessorMixin) -> None:
        super().__init__()
        self.name = name
        self.processor = processor
        self.instances = []
        dataset = instantiate(DATASETS[name], _partial_=True)(tokenizer=None, data_args=None)
        instance_list = dataset.instances

        for index, instance in enumerate(instance_list):
            messages = dataset.process(instance)
            media = extract_media(messages, draft=True)
            for k, image in enumerate(media.get("image", [])):
                info = {"uid": f"{name}/{index}-{k}", "image": image}
                info["value"] = messages
                self.instances.append(info)
        logger.info(f"[{name}] Loading total {len(self.instances)} images")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        instance = self.instances[index]
        uid, path = instance["uid"], instance["image"].path
        _image = Image.open(path)
        image = _image.convert("RGB")
        width, height = image.size
        if width < 10:
            image = add_margin(image, right=10)
        if height < 10:
            image = add_margin(image, bottom=10)
        image = self.processor(images=[image], return_tensors="pt").pixel_values[0]
        full_text = "\n######\n".join([txt["value"] for txt in instance["value"]])
        return {"uid": uid, "path": path, "image": image, "text": instance["value"][0]["value"], "full_text": full_text}

    def __len__(self) -> int:
        return len(self.instances)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdb-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, default=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    # Set up distributed environment
    dist.init()
    torch.cuda.set_device(dist.local_rank())

    # Load model and processor
    model = AutoModel.from_pretrained(args.model_name_or_path).cuda()
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    for name in tqdm(parse_mixture(args.dataset), disable=not dist.is_main()):
        dataset = ImageDataset(name, processor)
        # dataset = torch.utils.data.Subset(dataset, range(100))
        if os.path.exists(os.path.join(args.mmdb_dir, name + ".jsonl")):
            logger.warning(f"Skipping '{name}' as it already exists.")
            continue

        # dataset = torch.utils.data.Subset(dataset, range(
        #     79 * 4096 + 512 * 17 + 8 * 240 + 20,
        #     len(dataset)
        # ))
        # Set up image dataset and data loader
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=False,
            drop_last=True,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
        )

        # Extract image features and metainfos
        features, text_features, metainfos = [], [], []
        for _idx1, batch in enumerate(tqdm(data_loader, disable=not dist.is_main(), leave=False)):
            # print(batch)
            # input()
            image = batch["image"].cuda(non_blocking=True)
            text = batch["text"]
            with torch.inference_mode(), torch.cuda.amp.autocast():
                img_feats = model.get_image_features(image)
                text_emd = tokenizer(text, padding=True, return_tensors="pt")
                # clip only supports max to 77 tokens
                # siglip only supports max to 64 tokens
                text_input_ids = text_emd["input_ids"][:, :64].cuda()
                text_len = text_input_ids.shape[1]
                text_pos_ids = torch.arange(text_len).repeat(text_input_ids.shape[0], 1).cuda()
                txt_feats = model.get_text_features(input_ids=text_input_ids, position_ids=text_pos_ids)

            features.append(img_feats.cpu())
            text_features.append(txt_feats.cpu())
            for _idx, (uid, path, full_text) in enumerate(zip(batch["uid"], batch["path"], batch["full_text"])):
                metainfos.append({"uid": uid, "path": path, "full_text": full_text})

        # Gather image features and metainfos from all ranks
        features = dist.gather(features, dst=0)
        metainfos = dist.gather(metainfos, dst=0)
        if not dist.is_main():
            continue
        features = list(chain(*features))
        metainfos = list(chain(*metainfos))

        # Save image features and metainfos
        if features and metainfos:
            io.save(os.path.join(args.mmdb_dir, name + ".pt"), torch.cat(features))
            io.save(os.path.join(args.mmdb_dir, name + "_text.pt"), torch.cat(text_features))
            io.save(os.path.join(args.mmdb_dir, name + ".jsonl"), metainfos)
        else:
            logger.warning(f"Skipping '{name}' as no image features are extracted.")

    # Save configuration
    if dist.is_main():
        config = {"model_name_or_path": args.model_name_or_path}
        io.save(os.path.join(args.mmdb_dir, "config.json"), config, indent=4)


if __name__ == "__main__":
    main()
