import os
import json

path_to_pred = "/home/ligengz/workspace/VILA-internal/vertex-ai-gemini-15_pexel_1k_new_prompt.json"
labels_json = json.load(open("/lustre/fs2/portfolios/nvr/users/yukangc/datasets/Video-Benchmark-Label-0605.json"))

videos = []
preds_dict = {}
for item in preds_json:
    videos.append(item.split("/")[-1].split(".")[0])
    preds_dict[item.split("/")[-1].split(".")[0]] = preds_json[item]

model = "Gemini"
pred_path = "./eval_output/%s"%model

output_json = []
for item in labels_json:
    video_name = item['video_name']
    item_output = item
    item_output['pred'] = preds_dict[video_name]['output']
    output_json.append(item_output)

json.dump(output_json, open(os.path.join(pred_path, "pred.json"), "w"))
