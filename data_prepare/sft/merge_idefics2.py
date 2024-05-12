import os
import json

dataset_path = "/home/jasonlu/workspace/idefics2-sft/the_cauldron"
save_path = "/home/jasonlu/workspace/idefics2-sft/new-vflan/"
metadata_path = os.path.join(save_path, "metadata")
dataset_names = sorted(os.listdir(metadata_path))

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

all_data = []
for dataset_name in dataset_names:
    if "websight" in dataset_name or "datikz" in dataset_name:
        # skip the snapshot => code datasets for now.
        continue
    loaded = load_jsonl(os.path.join(metadata_path, dataset_name))
    id_offset = len(all_data)
    for item in loaded:
        item["id"] += id_offset
    all_data += loaded
    print(dataset_name, len(all_data), all_data[-1])

with open(os.path.join(save_path, "idefics2_sft_train.jsonl"), "w") as f:
    for item in all_data:
        json.dump(item, f)
        f.write("\n")