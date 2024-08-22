# This file is originated from the official MMMU codebase:
# https://github.com/MMMU-Benchmark/MMMU

import itertools
import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.eval.mmmu_utils.data_utils import CAT_SHORT2LONG, DOMAIN_CAT2SUB_CAT, construct_prompt, process_single_sample
from llava.eval.mmmu_utils.eval_utils import calculate_ins_level_acc, evaluate, parse_choice, parse_open_response
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger

CONFIG = {
    "task_instructions": "",
    "multi_choice_example_format": """{}

{}

Answer with the option's letter from the given choices directly.""",
    "short_ans_example_format": """{}

Answer the question using a single word or phrase.""",
    "temperature": 0,
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_mmmu(output_dict, answer_dict):
    # group by category
    output_dict_w_cat = {}
    for data_id, parsed_pred in output_dict.items():
        category = "_".join(data_id.split("_")[1:-1])
        if category not in output_dict_w_cat:
            output_dict_w_cat.update({category: {}})
        output_dict_w_cat[category].update({data_id: parsed_pred})

    # group by category
    answer_dict_w_cat = {}
    for data_id, parsed_pred in answer_dict.items():
        category = "_".join(data_id.split("_")[1:-1])
        if category not in answer_dict_w_cat:
            answer_dict_w_cat.update({category: {}})
        answer_dict_w_cat[category].update({data_id: parsed_pred})

    evaluation_result = {}

    for category in CAT_SHORT2LONG.values():
        print(f"Evaluating: {category}")
        # get cat_outputs and cat_answers
        try:
            cat_outputs = output_dict_w_cat[category]
            cat_answers = answer_dict_w_cat[category]
        except KeyError:
            print(f"Skipping {category} for not found")
            continue

        exampels_to_eval = []
        for data_id, parsed_pred in cat_outputs.items():
            question_type = cat_answers[data_id]["question_type"]
            if question_type != "multiple-choice":
                parsed_pred = parse_open_response(parsed_pred)  # mainly for type consistency (make it number, etc.)
            else:
                parsed_pred = parsed_pred

            exampels_to_eval.append(
                {
                    "id": data_id,
                    "question_type": question_type,
                    "answer": cat_answers[data_id]["ground_truth"],
                    "parsed_pred": parsed_pred,
                }
            )

        judge_dict, metric_dict = evaluate(exampels_to_eval)
        metric_dict.update({"num_example": len(exampels_to_eval)})

        evaluation_result[category] = metric_dict

    printable_results = {}
    # pdb.set_trace()
    # add domain Subject
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:  # use the order in DOMAIN_CAT2SUB_CAT
            if cat_name in evaluation_result.keys():
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum([cat_results["num_example"] for cat_results in in_domain_cat_results.values()])
        printable_results["Overall-" + domain] = {"num": int(in_domain_data_num), "acc": round(in_domain_ins_acc, 3)}
        # add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {"num": int(cat_results["num_example"]), "acc": round(cat_results["acc"], 3)}

    # table.append(["-----------------------------", "-----", "----"])
    all_ins_acc = calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 3),
    }

    print(printable_results)
    return printable_results


def main():
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--answer-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # TODO(zhijianl): Is this necessary?
    set_seed(42)

    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # TODO(zhijianl): This will be removed in the future
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    # Load model
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)

    # Set up generation config
    generation_config = model.generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    # Load data and chunk it
    split = "validation" if args.split == "val" else "test"
    data = concatenate_datasets([load_dataset(args.data_path, name, split=split) for name in CAT_SHORT2LONG.values()])
    instances = data.select(range(dist.rank(), len(data), dist.size()))

    # Run inference
    outputs = {}
    for instance in tqdm(instances, disable=not dist.is_main()):
        instance = process_single_sample(instance)
        instance = construct_prompt(instance, CONFIG)

        images = instance["image"]
        prompt = instance["final_input_prompt"].replace("<image>", "").strip()

        response = model.generate_content(images + [prompt], generation_config=generation_config)
        if instance["question_type"] == "multiple-choice":
            response = parse_choice(response, instance["all_choices"], instance["index2ans"])
        outputs[instance["id"]] = response

    # Gather and save outputs
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = dict(itertools.chain(*[output.items() for output in outputs]))
    io.save(os.path.join(args.output_dir, "outputs.json"), outputs)

    # Run evaluation
    if args.split == "val":
        answers = io.load(args.answer_path)
        metrics = evaluate_mmmu(outputs, answers)
        io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
        logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
