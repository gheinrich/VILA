#!/bin/bash
MODEL_PATH=$1
CKPT=$2

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.evaluate_vqa \
    --model-path $MODEL_PATH \
    --image-folder ./playground/data/eval/ai2d \
    --dataset ai2diagram_test \
    --conv-mode vicuna_v1