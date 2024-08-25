import json
import os.path as osp

import fire


def main(answer_file, output_file=None):
    jinfo = json.load(open("/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/Video-MME/qa_old_format.json"))
    labeled_key = {}
    if osp.exists(answer_file):
        labeled_key = json.load(open(answer_file))
    print(f"[{answer_file}] already answered ", len(labeled_key.keys()))

    for vmeta in jinfo:
        for question in vmeta["questions"]:
            qid = question["question_id"]
            if qid in labeled_key:
                # question["response"] = labeled_key[qid]["response"]
                question["response_w/o_sub"] = labeled_key[qid]["response_w/o_sub"]
                question["response_w/_sub"] = labeled_key[qid]["response_w/_sub"]
            else:
                # if not answered, using "C" as the default answer.
                print("missing", qid)
                question["response_w/o_sub"] = "C"
                question["response_w/_sub"] = "C"

    if output_file is None:
        output_file = answer_file.replace(".json", "_converted.json")
    with open(output_file, "w") as fp:
        json.dump(jinfo, fp, indent=2)
    return 0


if __name__ == "__main__":
    fire.Fire(main)
