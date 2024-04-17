#!/bin/bash
MODEL_PATH=$1
CKPT=$2

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.evaluate_vqa \
    --model-path $MODEL_PATH \
    --image-folder ./playground/data/eval/chartqa \
    --dataset  chartqa_test_human \
    --conv-mode vicuna_v1 \
    --answers-file ./eval_output/$CKPT/chartqa/answers1/merge.jsonl

python -m llava.eval.evaluate_vqa_score --answers-file ./eval_output/$CKPT/chartqa/answers1/merge.jsonl --dataset  chartqa_test_human

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.evaluate_vqa \
    --model-path $MODEL_PATH \
    --image-folder ./playground/data/eval/chartqa \
    --dataset  chartqa_test_augmented \
    --conv-mode vicuna_v1 \
    --answers-file ./eval_output/$CKPT/chartqa/answers2/merge.jsonl

python -m llava.eval.evaluate_vqa_score --answers-file --answers-file ./eval_output/$CKPT/chartqa/answers2/merge.jsonl --dataset  chartqa_test_human