#!/bin/bash
MODEL_PATH=$1
CKPT=$2

python -m llava.eval.evaluate_vqa \
    --model-path $MODEL_PATH \
    --image-folder ./playground/data/eval/chartqa \
    --dataset  chartqa_test_human \
    --conv-mode vicuna_v1

python -m llava.eval.evaluate_vqa \
    --model-path $MODEL_PATH \
    --image-folder ./playground/data/eval/chartqa \
    --dataset  chartqa_test_augmented \
    --conv-mode vicuna_v1