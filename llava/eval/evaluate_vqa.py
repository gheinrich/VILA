import argparse
import json
import os
import time
from typing import Optional

from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image


ds_collections = {
    'docvqa_test': {
        'test': './playground/data/eval/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'test': './playground/data/eval/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'test': './playground/data/eval/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_val': {
        'test': './playground/data/eval/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'test': './playground/data/eval/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'test': './playground/data/eval/ai2d/test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    }
}

# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--answer-dir", type=str, default="")
    args = parser.parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base,)

    questions = [json.loads(q) for q in open(os.path.expanduser(ds_collections[args.dataset]['test']), "r")]
    outputs = []
    for line in tqdm(questions):
        qs = line["question"]
        image_file = line['image']
        question_id = line['question_id']
        annotation = line['answer']
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        pred = model.generate(
            input_ids=input_ids.cuda(),
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=ds_collections[args.dataset]['max_new_tokens'],
            num_return_sequences=1,
            use_cache=True
        )
        answers = [
            tokenizer.decode(_[input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in pred
        ]

        answer = answers[0]
        if args.dataset in ['ocrvqa_val', 'ocrvqa_test']:
            outputs.append({
                'questionId': question_id,
                'answer': answer,
                'annotation': annotation,
            })
        elif args.dataset in ['ai2diagram_test']:
            outputs.append({
                'image': question_id,
                'answer': answer,
                'annotation': annotation,
            })
        elif args.dataset in ['chartqa_test_human', 'chartqa_test_augmented']:
            outputs.append({
                'answer': answer,
                'annotation': annotation,
            })
        elif args.dataset in ['docvqa_test']:
            outputs.append({
                'questionId': question_id,
                'answer': answer,
            })
        else:
            raise NotImplementedError

    print(f"Evaluating {args.dataset} ...")
    time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
    results_file = os.path.join(args.answer_dir, f'{args.dataset}_{time_prefix}.json')
    json.dump(outputs, open(results_file, 'w'), ensure_ascii=False)

    if ds_collections[args.dataset]['metric'] == 'relaxed_accuracy':
        print({
            'relaxed_accuracy': evaluate_relaxed_accuracy(outputs)
        })
    elif ds_collections[args.dataset]['metric'] == 'accuracy':
        if 'gqa' in args.dataset:
            for entry in outputs:
                response = entry['answer']
                response = response.strip().split('.')[0].split(
                    ',')[0].split('!')[0].lower()
                if 'is ' in response:
                    response = response.split('is ')[1]
                if 'are ' in response:
                    response = response.split('are ')[1]
                if 'a ' in response:
                    response = response.split('a ')[1]
                if 'an ' in response:
                    response = response.split('an ')[1]
                if 'the ' in response:
                    response = response.split('the ')[1]
                if ' of' in response:
                    response = response.split(' of')[0]
                response = response.strip()
                entry['answer'] = response
        print({'accuracy': evaluate_exact_match_accuracy(outputs)})