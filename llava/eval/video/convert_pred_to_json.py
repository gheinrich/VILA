import os
import json

labels_json = json.load(open("/lustre/fs2/portfolios/nvr/users/yukangc/datasets/Video-Benchmark-Label-0607.json"))

videos = [item['video_name'] for item in labels_json]

model = "VILA1.5-40B-shortvideos"
pred_path = "./eval_output/%s/Demo_Zero_Shot_QA_eval2/%s_bin_20_80"%(model, model)

pred_dict = {}
for pred in open(os.path.join(pred_path, "detailed_captions.txt")):
    if ".mp4: " in pred:
        video_name = pred.split(".mp4: ")[0]
        if not video_name in videos:
            continue
        content = pred.split(".mp4: ")[1].rstrip("\n")
        pred_dict[video_name] = content
    else:
        pred_dict[video_name] = pred_dict[video_name] + "\n%s"%pred

output_json = []
for item in labels_json:
    video_name = item["video_name"]
    item_output = item
    item_output['pred'] = pred_dict[video_name]
    output_json.append(item_output)

json.dump(output_json, open(os.path.join(pred_path, "pred.json"), "w"))
