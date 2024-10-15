import json

input_json_path = "/home/jasonlu/vlm_datasets3/estvqa/estvqa.json"
jsonl_path = "/home/jasonlu/vlm_datasets3/estvqa/ESTVQA_processed.jsonl"

with open(input_json_path) as f:
    data = json.load(f)

with open(jsonl_path, "w") as jsonl_file:
    for item in data:
        image_name = item["image"]
        convs = []
        for annotation in item["annotation"]:
            convs.extend(
                [{"from": "human", "value": annotation["question"]}, {"from": "gpt", "value": annotation["answer"]}]
            )

        convs[0]["value"] = "<image>\n" + convs[0]["value"]
        output = {"id": item["id"], "image": image_name, "conversations": convs}
        json.dump(output, jsonl_file, ensure_ascii=False)
        jsonl_file.write("\n")

print("Processing complete.")
