import argparse
import json
import os
import uuid
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from open_flamingo.eval.coco_metric import compute_cider
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    StoppingCriteria,
)

from llava.conversation import SeparatorStyle, conv_templates
from llava.eval.eval_utils import (
    CaptionDataset,
    HatefulMemesDataset,
    NoCapDataset,
    VQADataset,
    get_query_set,
    postprocess_ok_vqa_generation,
    postprocess_vqa_generation,
    prepare_eval_samples,
    sample_batch_demos_from_query_set,
)
from llava.eval.utils import RGBCheck, ToRGB, build_model, preprocess_image
from llava.eval.vqa_metric import compute_vqa_accuracy
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria
from llava.model.visual_attn_scale import new_attention_forward
from llava.train.dataset import tokenizer_image_token
from llava.train.token_config import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)

transformers.models.llama.modeling_llama.LlamaAttention.forward = new_attention_forward


def evaluate_captioning(
    args: argparse.Namespace,
    seed: int = 42,
    max_generation_length: int = 20,
    length_penalty: float = -2.0,
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 20.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
    Returns:
        float: CIDEr score

    """
    if args.dataset_name == "nocap":
        dataset = NoCapDataset(
            image_path="/tmp/images",
            json_path="/home/jil/datasets/nocap/nocaps_val_4500_captions.json",
        )
        generator = torch.Generator().manual_seed(42)
        from torch.utils.data import random_split

        train_dataset, test_dataset = random_split(
            dataset,
            [len(dataset) - args.num_samples, args.num_samples],
            generator=generator,
        )
    else:
        if args.dataset_name == "coco":
            image_train_dir_path = args.coco_train_image_dir_path
            image_val_dir_path = args.coco_val_image_dir_path
            annotations_path = args.coco_karpathy_json_path
        elif args.dataset_name == "flickr":
            image_train_dir_path = (
                args.flickr_image_dir_path
            )  # Note: calling this "train" for consistency with COCO but Flickr only has one split for images
            image_val_dir_path = None
            annotations_path = args.flickr_karpathy_json_path
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset_name}")

        train_dataset = CaptionDataset(
            image_train_dir_path=image_train_dir_path,
            image_val_dir_path=image_val_dir_path,
            annotations_path=annotations_path,
            is_train=True,
            dataset_name=args.dataset_name,
        )

        test_dataset = CaptionDataset(
            image_train_dir_path=image_train_dir_path,
            image_val_dir_path=image_val_dir_path,
            annotations_path=annotations_path,
            is_train=False,
            dataset_name=args.dataset_name,
        )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)

    test_dataset = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        seed,
    )
    (
        model,
        tokenizer,
        image_processor,
        conv_template,
        image_tokens,
        image_token_len,
    ) = build_model(args.model_name, args.conv_version)

    if (
        args.conv_version == "caption_simple"
    ):  # we have already added the instruction in the system part
        qs = image_tokens
    else:
        qs = image_tokens + "Give a short and clear explanation of the image."
        # qs = image_tokens + "Write a short description for the image."

    # now start evaluation
    predictions = defaultdict()

    # TODO: add batch size > 1 support
    for sample in tqdm(
        test_dataset, desc=f"Running inference {args.dataset_name.upper()}"
    ):
        # sample: "image", "caption", "image_id"
        conv = conv_template.copy()
        image_list = []
        if args.num_shots > 0:  # currently it is only trained with bs = 1
            demo_samples = sample_batch_demos_from_query_set(
                in_context_samples, args.num_shots, 1
            )[0]
            assert len(demo_samples) == args.num_shots
            for d_sample in demo_samples:
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], d_sample["caption"])
                image_list.append(d_sample["image"])
        else:
            demo_samples = sample_batch_demos_from_query_set(in_context_samples, 2, 1)[
                0
            ]
            assert len(demo_samples) == 2
            for d_sample in demo_samples:
                conv.append_message(conv.roles[0], qs.replace(image_tokens, ""))
                conv.append_message(conv.roles[1], d_sample["caption"])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        image_list.append(sample["image"])

        prompt = conv.get_prompt()

        # inputs = tokenizer([prompt])
        # input_ids = torch.as_tensor(inputs.input_ids).cuda()
        input_ids = (
            tokenizer_image_token(
                prompt,
                tokenizer,
                n_image_tokens=image_token_len,
                image_token_index=32000,
                return_tensors="pt",
            )
            .cuda()
            .view(1, -1)
        )

        image_tensor = torch.cat(
            [
                preprocess_image(
                    img, image_processor, use_padding="pad" in args.model_name
                )
                for img in image_list
            ],
            dim=0,
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(),
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=max_generation_length,
                    num_beams=args.num_beams,
                    # length_penalty=length_penalty,
                    stopping_criteria=[stopping_criteria],
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                (input_ids != output_ids[:, :input_token_len]).sum().item()
            )
            if n_diff_input_output > 0:
                print(
                    f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
        except Exception as e:
            print("#" * 20)
            print("Exception:", e)
            outputs = "none"
        if args.verbose:
            print(sample["image_id"], outputs)

        predictions[sample["image_id"]] = {
            "caption": outputs,
        }

    # save the predictions to a temporary file
    results_path = f"{args.dataset_name}results_{uuid.uuid4()}.json"

    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": predictions[k]["caption"]}
                    for k in predictions
                ],
                indent=4,
            )
        )
    annotation_path = {
        "coco": args.coco_annotations_json_path,
        "flickr": args.flickr_annotations_json_path,
        "nocap": "/home/jil/datasets/nocap/nocaps_val_4500_captions.json",
    }
    metrics = compute_cider(
        result_path=results_path,
        annotations_path=annotation_path[args.dataset_name],
    )

    # delete the temporary file
    os.remove(results_path)

    print("*" * 20)
    print(f"Results of {args.num_shots}:", metrics)
    print("*" * 20)

    return metrics["CIDEr"] * 100.0


def _get_vqa_path(args):
    if args.dataset_name == "okvqa":
        train_image_dir_path = args.ok_vqa_train_image_dir_path
        train_questions_json_path = args.ok_vqa_train_questions_json_path
        train_annotations_json_path = args.ok_vqa_train_annotations_json_path
        test_image_dir_path = args.ok_vqa_test_image_dir_path
        test_questions_json_path = args.ok_vqa_test_questions_json_path
        test_annotations_json_path = args.ok_vqa_test_annotations_json_path
    elif args.dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
    elif args.dataset_name == "vizwiz":
        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path
    elif args.dataset_name == "textvqa":
        train_image_dir_path = args.textvqa_image_dir_path
        train_questions_json_path = args.textvqa_train_questions_json_path
        train_annotations_json_path = args.textvqa_train_annotations_json_path
        test_image_dir_path = args.textvqa_image_dir_path
        test_questions_json_path = args.textvqa_test_questions_json_path
        test_annotations_json_path = args.textvqa_test_annotations_json_path
    elif args.dataset_name == "vqav2":
        train_image_dir_path = args.vqav2_train_image_dir_path
        train_questions_json_path = args.vqav2_train_questions_json_path
        train_annotations_json_path = args.vqav2_train_annotations_json_path
        test_image_dir_path = args.vqav2_test_image_dir_path
        test_questions_json_path = args.vqav2_test_questions_json_path
        test_annotations_json_path = args.vqav2_test_annotations_json_path
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    return (
        train_image_dir_path,
        train_questions_json_path,
        train_annotations_json_path,
        test_image_dir_path,
        test_questions_json_path,
        test_annotations_json_path,
    )


def evaluate_vqa(
    args: argparse.Namespace,
    seed: int = 42,
    max_generation_length: int = 5,
    length_penalty: float = -2.0,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """
    (
        train_image_dir_path,
        train_questions_json_path,
        train_annotations_json_path,
        test_image_dir_path,
        test_questions_json_path,
        test_annotations_json_path,
    ) = _get_vqa_path(args)

    train_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=args.dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=args.dataset_name,
    )

    (
        model,
        tokenizer,
        image_processor,
        conv_template,
        image_tokens,
        image_token_len,
    ) = build_model(args.model_name, args.conv_version)

    test_dataset = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        seed,
    )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)
    predictions = []

    if args.ocr_tokens:
        train_ocr_json_path = (
            "/home/jil/datasets/textvqa/TextVQA_Rosetta_OCR_v0.2_train.json"
        )
        test_ocr_json_path = (
            "/home/jil/datasets/textvqa/TextVQA_Rosetta_OCR_v0.2_val.json"
        )
        with open(train_ocr_json_path) as f:
            train_ocr = json.load(f)["data"]
            train_ocr = {d["image_id"]: d["ocr_tokens"] for d in train_ocr}
        with open(test_ocr_json_path) as f:
            test_ocr = json.load(f)["data"]
            test_ocr = {d["image_id"]: d["ocr_tokens"] for d in test_ocr}

    # get llava-1.5 prompt is applicable
    if args.llava15:
        if args.dataset_name in ["textvqa", "vqav2"]:
            llava15_prompt_suffix = (
                "\nAnswer the question using a single word or phrase."
            )
        elif args.dataset_name in ["vizwiz"]:
            llava15_prompt_suffix = "\nWhen the provided information is insufficient, respond with 'Unanswerable'. Answer the question using a single word or phrase."
            llava15_prompt_suffix = (  # Do not add
                "\nAnswer the question using a single word or phrase."
            )
        else:
            raise NotImplemented
    else:
        llava15_prompt_suffix = ""

    # my own acc calculation
    n_total = 0
    n_correct = 0
    for sample in tqdm(
        test_dataset, desc=f"Running inference {args.dataset_name.upper()}"
    ):
        # sample: "image", "question", "answers" [], "question_id"
        conv = conv_template.copy()
        image_list = []

        if (
            args.llava15 and args.num_shots == 0
        ):  # actually no sample for 0-shot in llava15
            demo_samples = []
        else:
            demo_samples = sample_batch_demos_from_query_set(
                in_context_samples, args.num_shots or 2, 1
            )[0]
        assert len(demo_samples) == args.num_shots or 2
        for d_sample in demo_samples:
            if args.ocr_tokens:
                tokens = train_ocr[d_sample["image_id"]]
                ocr_tokens = "OCR tokens: {}\n".format(", ".join(tokens))
            else:
                ocr_tokens = ""
            conv.append_message(
                conv.roles[0],
                (image_tokens if args.num_shots else "")
                + ocr_tokens
                + d_sample["question"]
                + llava15_prompt_suffix,
            )
            conv.append_message(conv.roles[1], d_sample["answers"][0])
            if args.num_shots:
                image_list.append(d_sample["image"])

        if args.ocr_tokens:
            tokens = test_ocr[sample["image_id"]]
            ocr_tokens = "OCR tokens: {}\n".format(", ".join(tokens))
        else:
            ocr_tokens = ""
        conv.append_message(
            conv.roles[0],
            image_tokens + ocr_tokens + sample["question"] + llava15_prompt_suffix,
        )
        conv.append_message(conv.roles[1], None)
        image_list.append(sample["image"])

        prompt = conv.get_prompt()

        # inputs = tokenizer([prompt])
        # input_ids = torch.as_tensor(inputs.input_ids).cuda()
        input_ids = (
            tokenizer_image_token(
                prompt,
                tokenizer,
                n_image_tokens=image_token_len,
                image_token_index=32000,
                return_tensors="pt",
            )
            .cuda()
            .view(1, -1)
        )

        image_tensor = torch.cat(
            [
                preprocess_image(
                    img, image_processor, use_padding="pad" in args.model_name
                )
                for img in image_list
            ],
            dim=0,
        )

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        n_total += 1
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(),
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=max_generation_length,
                    num_beams=args.num_beams,
                    # length_penalty=length_penalty,
                    stopping_criteria=[stopping_criteria],
                )

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]

            if args.verbose:
                print(
                    sample["question_id"],
                    sample["question"],
                    sample["answers"],
                    outputs,
                )

            if isinstance(sample["answers"], list):
                if outputs.lower() in [a.lower() for a in sample["answers"]]:
                    n_correct += 1
            else:
                if outputs.lower() == sample["answers"].lower():
                    n_correct += 1

            process_function = (
                postprocess_ok_vqa_generation
                if args.dataset_name == "okvqa"
                else postprocess_vqa_generation
            )

            outputs = process_function(outputs)
        except Exception as e:
            print("#" * 20)
            print("Exception:", e)
            outputs = "none"

        predictions.append({"answer": outputs, "question_id": sample["question_id"]})

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"{args.dataset_name}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"{args.dataset_name}results_{random_uuid}.json",
        test_questions_json_path,
        test_annotations_json_path,
    )

    # delete the temporary file
    os.remove(f"{args.dataset_name}results_{random_uuid}.json")

    print("*" * 20)
    print(f"Accuracy of {args.num_shots}:", acc)
    print(f"fast acc", n_correct / n_total)
    print("*" * 20)

    return acc


def evaluate_vqa_rank(
    args: argparse.Namespace,
    seed: int = 42,
    max_generation_length: int = 5,
    length_penalty: float = -2.0,
):
    """
    Evaluate a model on VQA datasets. Currently supports VQA v2.0, OK-VQA, VizWiz and TextVQA.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        max_generation_length (int, optional): max generation length. Defaults to 5.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        dataset_name (string): type of vqa dataset: currently supports vqav2, ok_vqa. Defaults to vqav2.
    Returns:
        float: accuracy score
    """

    (
        train_image_dir_path,
        train_questions_json_path,
        train_annotations_json_path,
        test_image_dir_path,
        test_questions_json_path,
        test_annotations_json_path,
    ) = _get_vqa_path(args)

    if args.dataset_name == "okvqa":
        candidate_path = "/home/jil/workspace/LLaVA/data/okvqa_candidates.json"
    else:
        raise NotImplementedError

    with open(candidate_path) as f:
        candidates = json.load(f)
        candidates = [c for c in candidates if c != ""]
        print(len(candidates))

    train_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=args.dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=args.dataset_name,
    )

    (
        model,
        tokenizer,
        image_processor,
        conv_template,
        image_tokens,
        image_token_len,
    ) = build_model(args.model_name, args.conv_version)

    test_dataset = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        seed,
    )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)
    predictions = []

    use_kv_cache = False

    if use_kv_cache:
        candidate_batch_size = 1024
        candidates = [" " + c + "</s>" for c in candidates]
        candidates_encoded = []
        for i in range(0, len(candidates), candidate_batch_size):
            tokenized = tokenizer(
                candidates[i : i + candidate_batch_size],
                padding="longest",
                return_tensors="pt",
            ).input_ids
            candidates_encoded.append(tokenized[:, 1:].cuda())
        # candidates_encoded = [tokenizer([" " + c + "</s>"], return_tensors="pt").input_ids[:, 1:].cuda() for c in candidates]

    for sample in tqdm(
        test_dataset, desc=f"Running inference {args.dataset_name.upper()}"
    ):
        # sample: "image", "question", "answers" [], "question_id"
        conv = conv_template.copy()
        image_list = []
        if args.num_shots > 0:  # currently it is only trained with bs = 1
            demo_samples = sample_batch_demos_from_query_set(
                in_context_samples, args.num_shots, 1
            )[0]
            assert len(demo_samples) == args.num_shots
            for d_sample in demo_samples:
                conv.append_message(conv.roles[0], image_tokens + d_sample["question"])
                conv.append_message(conv.roles[1], d_sample["answers"][0])
                image_list.append(d_sample["image"])
        else:
            demo_samples = sample_batch_demos_from_query_set(in_context_samples, 2, 1)[
                0
            ]
            assert len(demo_samples) == 2
            for d_sample in demo_samples:
                conv.append_message(conv.roles[0], d_sample["question"])
                conv.append_message(conv.roles[1], d_sample["answers"][0])

        conv.append_message(conv.roles[0], image_tokens + sample["question"])
        conv.append_message(conv.roles[1], None)
        image_list.append(sample["image"])

        prompt = conv.get_prompt()

        # prepend the space to the last
        if not use_kv_cache:
            prompt += " "
        input_ids = (
            tokenizer_image_token(
                prompt,
                tokenizer,
                n_image_tokens=image_token_len,
                image_token_index=32000,
                return_tensors="pt",
            )
            .cuda()
            .view(1, -1)
        )

        image_tensor = (
            torch.cat(
                [
                    preprocess_image(
                        img, image_processor, use_padding="pad" in args.model_name
                    )
                    for img in image_list
                ],
                dim=0,
            )
            .half()
            .cuda()
        )

        loss_list = []

        with torch.inference_mode():
            if use_kv_cache:
                outputs = model(input_ids, images=image_tensor, use_cache=True)
                past_key_values = outputs.past_key_values

                for candidate in candidates_encoded:
                    labels = candidate.clone()
                    labels[labels == tokenizer.pad_token_id] = -100
                    logits = model(
                        candidate, past_key_values=past_key_values, labels=labels
                    ).logits

                    for ii in range(logits.shape[0]):
                        cur_loss = torch.nn.CrossEntropyLoss()(
                            logits[ii, :-1],
                            labels[ii, 1:],
                        ).item()
                        # whether to normalize?
                        loss_list.append(cur_loss)

            else:
                # firstly, sample 128 most likely candidates according to the first token
                answer_ids = tokenizer(
                    candidates, padding="longest", return_tensors="pt"
                ).input_ids.cuda()
                start_output = model(input_ids, images=image_tensor)
                logits = start_output.logits[:, -1, :]

                answer_first_token = answer_ids[:, 1]
                assert len(logits.shape) == 2
                prob_first_token = F.softmax(logits, dim=1).index_select(
                    dim=1, index=answer_first_token
                )
                topk_probs, topk_ids = prob_first_token.topk(128, dim=1)
                topk_ids = topk_ids.view(-1).cpu().numpy().tolist()

                for answer_idx in topk_ids:
                    extra_ids = (
                        tokenizer(
                            [candidates[answer_idx] + "</s>"], return_tensors="pt"
                        )
                        .input_ids[:, 1:]
                        .cuda()
                    )

                    full_ids = torch.cat([input_ids, extra_ids], dim=1).view(1, -1)
                    target = full_ids.clone()
                    target[:, : input_ids.shape[-1]] = -100
                    target[target == tokenizer.pad_token_id] = -100
                    outputs = model(
                        full_ids,
                        images=image_tensor,
                        labels=target,
                    )
                loss_list.append(outputs.loss.item())

        outputs = candidates[topk_ids[np.argmin(loss_list)]].replace("</s>", "").strip()

        if args.verbose:
            print(sample["question_id"], sample["question"], sample["answers"], outputs)

        process_function = (
            postprocess_ok_vqa_generation
            if args.dataset_name == "okvqa"
            else postprocess_vqa_generation
        )

        outputs = process_function(outputs)

        predictions.append({"answer": outputs, "question_id": sample["question_id"]})

    # save the predictions to a temporary file
    random_uuid = str(uuid.uuid4())
    with open(f"{args.dataset_name}results_{random_uuid}.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = compute_vqa_accuracy(
        f"{args.dataset_name}results_{random_uuid}.json",
        test_questions_json_path,
        test_annotations_json_path,
    )

    # delete the temporary file
    os.remove(f"{args.dataset_name}results_{random_uuid}.json")

    print("*" * 20)
    print(f"Accuracy of {args.num_shots}:", acc)
    print("*" * 20)

    return acc


def evaluate_classification(
    args: argparse.Namespace,
    seed: int = 42,
    no_kv_caching=False,
):
    """
    Evaluate a model on classification dataset.

    Args:
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): random seed. Defaults to 42.
        num_shots (int, optional): number of shots to use. Defaults to 8.
        no_kv_caching (bool): whether to disable key-value caching
        dataset_name (str, optional): dataset name. Defaults to "imagenet".

    Returns:
        float: accuracy score
    """
    if args.dataset_name == "hateful_memes":
        train_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_train_annotations_json_path,
        )
        test_dataset = HatefulMemesDataset(
            args.hateful_memes_image_dir_path,
            args.hateful_memes_test_annotations_json_path,
        )
        all_class_names = ["no", "yes"]
        k = 1
        q_template = "A meme with: '{}' written on it. Is it hateful?"
    else:
        raise ValueError(f"Unsupported dataset {args.dataset_name}")

    class_id_to_name = dict(zip(range(len(all_class_names)), all_class_names))

    (
        model,
        tokenizer,
        image_processor,
        conv_template,
        image_tokens,
        image_token_len,
    ) = build_model(args.model_name, args.conv_version)

    test_dataset = prepare_eval_samples(
        test_dataset,
        args.num_samples if args.num_samples > 0 else len(test_dataset),
        seed,
    )

    in_context_samples = get_query_set(train_dataset, args.query_set_size, seed)

    predictions = []

    for sample in tqdm(
        test_dataset, desc=f"Running inference {args.dataset_name.upper()}"
    ):
        conv = conv_template.copy()
        image_list = []
        demo_samples = sample_batch_demos_from_query_set(
            in_context_samples, args.num_shots or 2, 1
        )[0]
        assert len(demo_samples) == args.num_shots or 2

        for d_sample in demo_samples:
            conv.append_message(
                conv.roles[0],
                (image_tokens if args.num_shots else "")
                + q_template.format(d_sample["ocr"]),
            )
            conv.append_message(conv.roles[1], d_sample["class_name"])
            if args.num_shots:
                image_list.append(d_sample["image"])

        conv.append_message(
            conv.roles[0], image_tokens + q_template.format(sample["ocr"])
        )
        conv.append_message(conv.roles[1], None)
        image_list.append(sample["image"])

        prompt = conv.get_prompt() + " "

        input_ids = (
            tokenizer_image_token(
                prompt,
                tokenizer,
                n_image_tokens=image_token_len,
                image_token_index=32000,
                return_tensors="pt",
            )
            .cuda()
            .view(1, -1)
        )
        image_tensor = (
            torch.cat(
                [
                    preprocess_image(
                        img, image_processor, use_padding="pad" in args.model_name
                    )
                    for img in image_list
                ],
                dim=0,
            )
            .half()
            .cuda()
        )

        overall_probs = []

        for class_name in all_class_names:
            classname_tokens = tokenizer(
                class_name, add_special_tokens=False, return_tensors="pt"
            ).input_ids.cuda()
            assert len(classname_tokens.shape) == 2

            with torch.inference_mode():
                full_input = torch.cat((input_ids, classname_tokens), dim=-1)
                logits = model(
                    full_input,
                    images=image_tensor,
                ).logits
                logprobs = torch.log_softmax(logits, dim=-1)

                gen_probs = logprobs[:, -classname_tokens.shape[-1] - 1 : -1, :]
                gen_probs = torch.gather(
                    gen_probs, 2, classname_tokens[:, :, None]
                ).squeeze(-1)

                normalize_length = True

                if normalize_length:
                    class_prob = torch.mean(gen_probs, dim=1)
                else:
                    class_prob = torch.sum(gen_probs, dim=1)
                overall_probs.append(class_prob)  # (B, 1)

        overall_probs = torch.vstack(overall_probs).T.cpu()  # 1, n
        pred_idx = overall_probs.view(-1).argmax().item()
        predicted_classnames = class_id_to_name[pred_idx]
        predicted_logprobs = overall_probs.view(-1)[pred_idx].item()

        # compute accuracy
        y_i = sample["class_name"]
        score = torch.exp(
            predicted_logprobs - torch.logsumexp(overall_probs.view(-1), dim=0)
        ).item()
        predictions.append(
            {
                "id": sample["id"],
                "gt_label": y_i,
                "pred_label": predicted_classnames,
                "pred_score": score,
            }
        )
    if args.dataset_name == "hateful_memes":
        # return ROC-AUC score
        greater_label = max(all_class_names)
        gts = [pred["gt_label"] for pred in predictions]
        pred_scores = [
            pred["pred_score"]
            if pred["pred_label"] == greater_label
            else 1 - pred["pred_score"]
            for pred in predictions
        ]
        from sklearn.metrics import roc_auc_score

        final_score = roc_auc_score(gts, pred_scores)
        print("*" * 20)
        print(f"ROC AUC of {args.num_shots}:", round(final_score, 3))
        print("*" * 20)
    else:
        # return top-1 accuracy
        acc1 = sum(int(pred["gt_label"] == pred["pred_label"]) for pred in predictions)
        return float(acc1) / len(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv_version", type=str, default="v1")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="coco")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--nll_rank", action="store_true")
    parser.add_argument("--ocr_tokens", action="store_true")

    parser.add_argument("--llava15", action="store_true")

    # data setting
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument(
        "--query_set_size",
        type=int,
        default=2048,
        help="Size of demonstration query set",
    )
    parser.add_argument("--num_shots", type=int, default=0)
    # COCO
    parser.add_argument(
        "--coco_train_image_dir_path", type=str, default="/tmp/coco/train2017/"
    )
    parser.add_argument(
        "--coco_val_image_dir_path", type=str, default="/tmp/coco/val2014/"
    )
    parser.add_argument(
        "--coco_annotations_json_path",
        type=str,
        default="/tmp/coco/annotations/captions_val2014.json",
    )
    parser.add_argument(
        "--coco_karpathy_json_path",
        type=str,
        default="/home/jil/datasets/karpathy_json/dataset_coco.json",
    )
    # Flickr
    parser.add_argument(
        "--flickr_image_dir_path", type=str, default="/tmp/flickr/flickr30k-images"
    )
    parser.add_argument(
        "--flickr_karpathy_json_path",
        type=str,
        default="/home/jil/datasets/karpathy_json/dataset_flickr30k.json",
    )
    parser.add_argument(
        "--flickr_annotations_json_path",
        type=str,
        default="/home/jil/datasets/flickr30k/flickr30k_coco_all.json",
    )
    # VQAv2
    parser.add_argument(
        "--vqav2_train_image_dir_path", type=str, default="/tmp/coco/train2014"
    )
    parser.add_argument(
        "--vqav2_train_questions_json_path",
        type=str,
        default="/home/jil/datasets/vqav2/v2_OpenEnded_mscoco_train2014_questions.json",
    )
    parser.add_argument(
        "--vqav2_train_annotations_json_path",
        type=str,
        default="/home/jil/datasets/vqav2/v2_mscoco_train2014_annotations.json",
    )
    parser.add_argument(
        "--vqav2_test_image_dir_path", type=str, default="/tmp/coco/val2014"
    )
    parser.add_argument(
        "--vqav2_test_questions_json_path",
        type=str,
        default="/home/jil/datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json",
    )
    parser.add_argument(
        "--vqav2_test_annotations_json_path",
        type=str,
        default="/home/jil/datasets/vqav2/v2_mscoco_val2014_annotations.json",
    )
    # OKVQA
    parser.add_argument(
        "--ok_vqa_train_image_dir_path", type=str, default="/tmp/coco/train2014"
    )
    parser.add_argument(
        "--ok_vqa_train_questions_json_path",
        type=str,
        default="/home/jil/datasets/okvqa/OpenEnded_mscoco_train2014_questions.json",
    )
    parser.add_argument(
        "--ok_vqa_train_annotations_json_path",
        type=str,
        default="/home/jil/datasets/okvqa/mscoco_train2014_annotations.json",
    )
    parser.add_argument(
        "--ok_vqa_test_image_dir_path", type=str, default="/tmp/coco/val2014"
    )
    parser.add_argument(
        "--ok_vqa_test_questions_json_path",
        type=str,
        default="/home/jil/datasets/okvqa/OpenEnded_mscoco_val2014_questions.json",
    )
    parser.add_argument(
        "--ok_vqa_test_annotations_json_path",
        type=str,
        default="/home/jil/datasets/okvqa/mscoco_val2014_annotations.json",
    )
    # TextVQA
    parser.add_argument(
        "--textvqa_image_dir_path", type=str, default="/tmp/textvqa/train_images"
    )
    parser.add_argument(
        "--textvqa_train_questions_json_path",
        type=str,
        default="/home/jil/datasets/textvqa/openflamingo/train_questions_vqa_format.json",
    )
    parser.add_argument(
        "--textvqa_train_annotations_json_path",
        type=str,
        default="/home/jil/datasets/textvqa/openflamingo/train_annotations_vqa_format.json",
    )
    parser.add_argument(
        "--textvqa_test_questions_json_path",
        type=str,
        default="/home/jil/datasets/textvqa/openflamingo/val_questions_vqa_format.json",
    )
    parser.add_argument(
        "--textvqa_test_annotations_json_path",
        type=str,
        default="/home/jil/datasets/textvqa/openflamingo/val_annotations_vqa_format.json",
    )
    # VizWiz
    parser.add_argument(
        "--vizwiz_train_image_dir_path", type=str, default="/tmp/vizwiz/train"
    )
    parser.add_argument(
        "--vizwiz_test_image_dir_path", type=str, default="/tmp/vizwiz/val"
    )
    parser.add_argument(
        "--vizwiz_train_questions_json_path",
        type=str,
        default="/home/jil/datasets/vizwiz/train_questions_vqa_format.json",
    )
    parser.add_argument(
        "--vizwiz_train_annotations_json_path",
        type=str,
        default="/home/jil/datasets/vizwiz/train_annotations_vqa_format.json",
    )
    parser.add_argument(
        "--vizwiz_test_questions_json_path",
        type=str,
        default="/home/jil/datasets/vizwiz/val_questions_vqa_format.json",
    )
    parser.add_argument(
        "--vizwiz_test_annotations_json_path",
        type=str,
        default="/home/jil/datasets/vizwiz/val_annotations_vqa_format.json",
    )
    ## Hateful Memes dataset
    parser.add_argument(
        "--hateful_memes_image_dir_path",
        type=str,
        default="/tmp/hateful_memes/hateful_memes/img",
    )
    parser.add_argument(
        "--hateful_memes_train_annotations_json_path",
        type=str,
        default="/tmp/hateful_memes/hateful_memes/train.jsonl",
    )
    parser.add_argument(
        "--hateful_memes_test_annotations_json_path",
        type=str,
        default="/tmp/hateful_memes/hateful_memes/dev_seen.jsonl",
    )

    args = parser.parse_args()

    if "336" in args.model_name and args.num_shots == 8 and not "ds" in args.model_name:
        print("context too long... skip")
        exit()

    if args.dataset_name in ["coco", "flickr", "nocap"]:
        evaluate_captioning(args)
    elif args.dataset_name in ["okvqa", "textvqa", "vizwiz", "vqav2"]:
        if args.nll_rank:
            evaluate_vqa_rank(args)
        else:
            evaluate_vqa(args)
    elif args.dataset_name in ["hateful_memes"]:
        evaluate_classification(args)
    else:
        raise NotImplementedError
