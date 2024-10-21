import argparse
import base64
import json
import os
import os.path as osp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# def fn(i):
#     sleep(1)
#     return i

# with ThreadPoolExecutor(max_workers=8) as executor:
#     futures = []
#     for i in range(100):
#         future = executor.submit(fn, i)
#         futures.append(future)
#     for future in as_completed(futures):
#         print(future.result())


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
    workers=1,
):
    # already captioned
    result = {}
    if osp.exists("data_curation_dev/recaptioned.json"):
        with open("data_curation_dev/recaptioned.json") as f:
            result = json.load(f)
        print("[recaptioned.json] loading ", len(list(result.values())), " labeled examples")

    features, text_features, metainfos = [], [], []
    for fpath in glob(os.path.join(mmdb_dir, "*.jsonl")):
        print("loading metainfos", fpath)
        metainfos.extend(io.load(fpath))

    indices_list = []
    for cosi_path in glob(os.path.join("data_curation_dev", "*.pth")):
        print("loading cosi", cosi_path)
        cosi = torch.load(cosi_path, map_location="cpu")
        topk_cosi, indices = torch.topk(cosi, k=topk)
        indices_list += indices.tolist()
    indices_list = sorted(list(set(indices_list)))
    print("total indices", len(indices_list))

    information = []
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
            # print("===" * 40)
            # print(uid, img_path, progress, len(indices_list))
            if uid in result:
                print("[already labeled], skip", uid, progress, len(indices_list))
                continue
            information.append(
                {
                    "uid": uid,
                    "img_path": img_path,
                    "question": question,
                    "answer": answer,
                    "prompt": prompt,
                    "meta": meta,
                }
            )

    print("job queue prefill ready", len(information))

    def labeling_function(info, model="gpt-4o"):
        uid = info["uid"]
        img_path = info["img_path"]
        prompt = info["prompt"]
        question = info["question"]
        answer = info["answer"]
        meta = info["meta"]

        try:
            text = describe_image(img_path, prompt=prompt, model=model)
        except Exception as e:
            return f"[error] {e}"
        if text is None:
            return f"[error] None response {uid} {img_path}"
        if "sorry" in text.lower() or "unable" in text.lower() or "apologize" in text.lower():
            return f"[weried sorry] skip {uid} {img_path}"

        labeled_res = meta
        # result["full_text"] = meta["full_text"].encode("utf-8").decode("utf-8")
        labeled_res["raw_question"] = question
        labeled_res["raw_answer"] = answer
        labeled_res["recaptioned_prompt"] = prompt
        labeled_res["recaptioned_answer"] = text
        labeled_res["labeler"] = model
        labeled_res["_uid"] = info["uid"]

        return labeled_res

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for info in information:
            future = executor.submit(labeling_function, info, model)
            futures.append(future)

        idx = -1
        for future in as_completed(futures):
            time.sleep(0.1)
            idx += 1
            if idx % 200 == 0 or idx == len(futures) - 1:
                with open("data_curation_dev/recaptioned.json", "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            print("===" * 40)
            print("[progress]", idx, len(futures))
            res = future.result()
            if isinstance(res, str):
                print(res)
                continue
            result[res["_uid"]] = res
            # print(res)
            print("labeling success", res["_uid"], res["path"])


if __name__ == "__main__":
    # Example usage
    import fire

    fire.Fire(label)
