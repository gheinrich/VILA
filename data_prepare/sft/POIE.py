import json
import os
import random
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm  # Add this import

entity_dict = {
    "SS": "Serving Size",
    "CE-PS": "Calories/Energy per serving",
    "CE-P1": "Calories per 100g/100ml",
    "CE-D": "Calories % Daily Value",
    "CE-PP": "Calories per package",
    "TF-PS": "Total Fat per serving",
    "TF-P1": "Total Fat per 100g/100ml",
    "TF-D": "Total Fat % Daily Value",
    "TF-PP": "Total Fat per package",
    "SO-PS": "Sodium per serving",
    "SO-P1": "Sodium per 100g/100ml",
    "SO-D": "Sodium % Daily Value",
    "SO-PP": "Sodium per package",
    "CAR-PS": "Total Carbohydrate per serving",
    "CAR-P1": "Total Carbohydrate per 100g/100ml",
    "CAR-D": "Total Carbohydrate % Daily Value",
    "CAR-PP": "Total Carbohydrate per package",
    "PRO-PS": "Protein per serving",
    "PRO-P1": "Protein per 100g/100ml",
    "PRO-D": "Protein % Daily Value",
    "PRO-PP": "Protein per package",
}

choice_1_prompts = [
    "Inside bounding box: {}, What is the text in the bounding box?",
    "What is the textual content enclosed by the coordinates {}?",
    "Inside the defined region {}, what text is present?",
    "What is the text contained within the rectangular area defined by the points {}?",
    "Can you identify the text that falls within the bounding box {}?",
    "What is written in the image inside the box {}?",
]

# PLEASE REPLACE YOUR IMAGE FOLDER HERE.
image_root = "/home/jasonlu/vlm_datasets3/poie/nfv5/nfv5_3125/image_files"
# PLEASE REPLACE YOUR ANNOTATION FILE HERE.
anns = []
with open("/home/jasonlu/vlm_datasets3/poie/nfv5/nfv5_3125/train.txt") as f:
    for line in f:
        anns.append(json.loads(line.strip()))

return_list = []

jsonl_path = os.path.join("/home/jasonlu/vlm_datasets3/poie/nfv5/nfv5_3125/", "POIE_processed.jsonl")


def coords_list2bbox(coords_list: List[List[int]], width: int, height: int) -> str:
    x = coords_list[0::2]
    y = coords_list[1::2]

    left = np.clip(int(min(x) / width * 1000), 0, 999)
    upper = np.clip(int(min(y) / height * 1000), 0, 999)
    right = np.clip(int(max(x) / width * 1000), 0, 999)
    bottom = np.clip(int(max(y) / height * 1000), 0, 999)

    return f"[{left:03d},{upper:03d},{right:03d},{bottom:03d}]"


# Add a progress bar
with open(jsonl_path, "w") as jsonl_file:
    for data in tqdm(anns, desc="Processing images", total=len(anns)):
        w = data["width"]
        h = data["height"]
        convs = []
        for v_i in data["annotations"]:
            coords = coords_list2bbox(v_i["polygon"], w, h)
            caption = v_i["text"]
            convs.extend(
                [
                    {
                        "from": "human",
                        "value": choice_1_prompts[random.randint(0, 4)].format(coords),
                    },
                    {"from": "gpt", "value": caption},
                ]
            )
        if "entity_dict" in data:
            for k, v in data["entity_dict"].items():
                convs.extend(
                    [
                        {"from": "human", "value": f"How much is {entity_dict[k]}?"},
                        {"from": "gpt", "value": v},
                    ]
                )
        if len(convs) > 0:
            convs[0]["value"] = "<image>\n" + convs[0]["value"]
            outputs = {
                "id": data["file_name"].split("/")[-1].split(".")[0],
                "image": data["file_name"],
                "conversations": convs,
            }
            json.dump(outputs, jsonl_file)
            jsonl_file.write("\n")  # Add a newline after each JSON object

print("Processing complete.")
