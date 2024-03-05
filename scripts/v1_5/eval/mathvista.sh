#!/bin/bash
MODEL_PATH=$1
CKPT=$2
SPLIT=$3

mkdir -p ./playground/data/eval/MathVista/answers/$CKPT

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.eval_mathvista \
    --model-path $MODEL_PATH \
    --split $SPLIT \
    --answers-file ./playground/data/eval/MathVista/answers/$CKPT/MathVista_$SPLIT.json \
    --temperature 0 \
    --conv-mode vicuna_v1