import json
import re
from collections import defaultdict
from pprint import pprint

f = json.load(open("data_curation_dev/recaptioned.json"))
new_info = {}
skip_reasons = defaultdict(int)


def contains_chinese(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def keywords_to_skip(text, key=("sorry", "unable", "apologize", "can't help", "cannot", "can't", "don't know")):
    for k in key:
        if k in text.lower():
            return True
    return False


for k, v in f.items():
    if keywords_to_skip(v["recaptioned_answer"]):
        # print("skip sorry", k)
        skip_reasons["sorry"] += 1
        continue
    if contains_chinese(v["raw_question"]) or contains_chinese(v["recaptioned_answer"]):
        # print("skip contains chinese", k)
        skip_reasons["contains_chinese"] += 1
        # pprint(v); input()
        continue

    new_k = "/".join(k.split("/")[:-1])
    if new_k not in new_info:
        new_info[new_k] = []

    question = v["raw_question"]
    ques = "?".join(question.split("?")[:-1]) + "?"
    if len(question.split("?")) == 1:
        ques = question
    prompt = (
        ques
        + " Now let's think step by step to explain the answer and each step shall be separated by '#####' at the beginning."
        + " In the last line, answer the question using a single word or phrase."
    )
    new_info[new_k].append(
        {
            "from": "human",
            "value": prompt,
        }
    )
    new_info[new_k].append(
        {
            "from": "gpt",
            "value": v["recaptioned_answer"],
        }
    )

print("filter stage1: ", len(new_info.keys()))
# input()


final_info = {}

for k, v in new_info.items():
    if len(v) < 2:
        print("no enough data", k)
        skip_reasons["no_enough_data"] += 1
        continue
    ########################################
    #######    filtering keywords    #######
    # if "doc" in k:
    #     print("skip doc", k)
    #     skip_reasons["doc"] += 1
    #     continue
    # if "sharegpt4v_sft" in k:
    #     print("skip sharegpt4v_sft", k)
    #     skip_reasons["sharegpt4v_sft"] += 1
    #     continue

    if "m4-instruct-video" in k:
        print("skip m4-instruct-video", k)
        skip_reasons["m4_instruct_video"] += 1
        continue

    if "#####" not in v[1]["value"]:
        print("skip missing #####", k)
        skip_reasons["missing_hashes"] += 1
        continue
    if "What scene is this picture depicting?" in v[0]["value"]:
        print("skip specific question", k)
        skip_reasons["specific_question_scene"] += 1
        continue
    if "Describe every detail in the picture." in v[0]["value"]:
        print("skip specific question", k)
        skip_reasons["specific_question_detail"] += 1
        continue
    if "[xmin, ymin, xmax, ymax]" in v[0]["value"]:
        print("skip bounding box", k)
        skip_reasons["bounding_box"] += 1
        continue
    if "sherlock" in k:
        print("skip sherlock", k)
        skip_reasons["sherlock"] += 1
        continue
    if "captioning_textcap_train" in k:
        print("skip captioning_textcap_train", k)
        skip_reasons["captioning_textcap_train"] += 1
        continue
    if "wit" in k:
        print("skip wit subset", k)
        skip_reasons["wit_subset"] += 1
        continue

    ########################################
    #######    dedup    #######
    conv = []
    question_set = set()
    for idx in range(0, len(v), 2):
        key = v[idx]["value"].lower()
        if key in question_set:
            print("dedup", k)
            skip_reasons["dedup"] += 1
            question_set.add(key)
            continue
        question_set.add(key)

        #   #     input()
        conv += [{"from": "human", "value": v[idx]["value"]}, {"from": "gpt", "value": v[idx + 1]["value"]}]
    if len(conv) > 2:
        final_info[k] = conv

pprint(skip_reasons)
pprint(len(final_info.keys()))
with open("data_curation_dev/recaptioned_cot_llava_1028.json", "w") as f:
    json.dump(final_info, f, indent=2, ensure_ascii=False)
