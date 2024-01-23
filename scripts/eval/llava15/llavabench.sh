#!/bin/bash
source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA/

python -m llava.eval.model_vqa \
    --model-name $1 \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1_1

mkdir -p $1/eval/llava15/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
    --output \
        $1/eval/llava15/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl

python llava/eval/summarize_gpt_review.py -f $1/eval/llava15/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl
