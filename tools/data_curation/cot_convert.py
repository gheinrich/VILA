import json

f = json.load(open("data_curation_dev/recaptioned.json"))

new_info = {}


def keywords_to_skip(text, key=("sorry", "unable", "apologize", "can't help")):
    for k in key:
        if k in text.lower():
            return True
    return False


for k, v in f.items():
    new_k = "/".join(k.split("/")[:-1])

    if new_k not in new_info:
        new_info[new_k] = []

    if keywords_to_skip(v["recaptioned_answer"]):
        print("skip", k)
        continue

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

final_info = {}
for k, v in new_info.items():
    if len(v) >= 1:
        final_info[k] = v

print(len(final_info.keys()))
with open("data_curation_dev/recaptioned_cot_llava_1021fixed.json", "w") as f:
    json.dump(final_info, f, indent=2, ensure_ascii=False)
