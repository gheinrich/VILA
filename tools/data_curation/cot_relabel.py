import argparse
import base64
import json
import os
import os.path as osp
from glob import glob
from io import BytesIO
from itertools import chain
from pprint import pprint
from typing import Any, Dict

import numpy as np
import openai
import requests
import torch
from datasets import load_dataset
from hydra.utils import instantiate
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from llava.utils import io

client = OpenAI()


def describe_image(image_url, model="gpt-4o-mini", prompt="describe the image briefly"):
    # Convert the image to PNG and save it as a temporary file
    image = Image.open(image_url)
    temp_file = BytesIO()
    image.save(temp_file, format="PNG")
    temp_file.seek(0)
    base64_image = base64.b64encode(temp_file.read()).decode("utf-8")

    # Call the OpenAI API to describe the image
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        # "image_url": f"data:image/jpeg;base64,{base64_image}"
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    # Extract and return the description
    return response.choices[0].message.content


def test():
    image_url = "demo_images/demo_img_1.png"
    description = describe_image(image_url)
    print(description)


task2fn = (
    "mmmu_val",
    "gqa",
    "docvqa",
    "chartqa",
    "mme",
    "ocrbench",
    "textvqa",
)


def label(
    mmdb_dir="data_curation_dev/mmdb",
    cosi_path="data_curation_dev/chartqa.pth",
    topk=1000,
    model="gpt-4o",
):
    features, text_features, metainfos = [], [], []
    for fpath in glob(os.path.join(mmdb_dir, "*.jsonl")):
        print("loading metainfos", fpath)
        metainfos.extend(io.load(fpath))

    result = {}
    if osp.exists("data_curation_dev/recaptioned.json"):
        with open("data_curation_dev/recaptioned.json") as f:
            result = json.load(f)
        print("loading ", len(list(result.values())), " labeled examples")

    indices_list = []
    for cosi_path in glob(os.path.join("data_curation_dev", "*.pth")):
        print("loading cosi", cosi_path)
        cosi = torch.load(cosi_path, map_location="cpu")
        topk_cosi, indices = torch.topk(cosi, k=topk)
        indices_list += indices.tolist()
    indices_list = sorted(list(set(indices_list)), reverse=True)

    for progress, idx in enumerate(indices_list):
        meta = metainfos[idx]
        uid = meta["uid"]

        qa_pairs = meta["full_text"].split("\n######\n")
        assert len(qa_pairs) % 2 == 0, f"Uneven QA pairs error: {uid}"

        for _idx in range(0, len(qa_pairs), 2):
            question = qa_pairs[_idx].strip()
            answer = qa_pairs[_idx + 1].strip()

            ques = "?".join(question.split("?")[:-1]) + "?"
            if len(question.split("?")) == 1:
                ques = question

            prompt = (
                ques
                + f" While the answer should be ({answer}) "
                + "now let's think step by step to explain the answer and each step shall be separated by '#####' at the beginning."
                + "In the last line, answer the question using a single word or phrase."
            )

            img_path = meta["path"]
            _uid = meta["uid"]
            uid = f"{_uid}/{_idx // 2}"
            print("===" * 40)
            print(uid, img_path, progress, len(indices_list))
            if uid in result:
                print("[already labeled], skip", uid)
                continue
            try:
                text = describe_image(img_path, prompt=prompt, model=model)
            except Exception as e:
                print("[error]", e)
                continue
            if "sorry" in text.lower() or "unable" in text.lower():
                print("[weried sorry] skip", uid, img_path, text)
                continue

            result[uid] = meta
            # result[uid]["full_text"] = meta["full_text"].encode("utf-8").decode("utf-8")
            result[uid]["raw_question"] = question
            result[uid]["raw_answer"] = answer
            result[uid]["recaptioned_prompt"] = prompt
            result[uid]["recaptioned_answer"] = text
            result[uid]["labeler"] = model

            print("---" * 40)
            meta["full_text"]
            print("---" * 40)
            print(prompt)
            print("---" * 40)
            print(text)

        with open("data_curation_dev/recaptioned.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    import fire

    fire.Fire(label)


"""
python tools/data_curation/cot_relabel.py --cosi_path=data_curation_dev/mmmu_val.pth
for f in data_curation_dev/*.pth; do
    echo $f
    python tools/data_curation/cot_relabel.py --cosi_path=$f
done

srun --account nvr_elm_llm \
    --partition cpu,cpu_long,cpu_short,cpu_interactive,interactive --job-name nvr_elm_llm:dev \
    --time 24:00:00 --exclusive python tools/data_curation/cot_relabel.py
"""
