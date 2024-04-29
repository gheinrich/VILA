#!/bin/bash
MODEL_PATH=$1
CKPT=$2
SPLIT=$3
CONV_MODE=vicuna_v1
if [ "$#" -ge 4 ]; then
    CONV_MODE="$4"
fi

mkdir -p ./playground/data/eval/MathVista/answers/$CKPT

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.eval_mathvista \
    --model-path $MODEL_PATH \
    --split $SPLIT \
    --answers-file ./eval_output/$CKPT/MathVista/MathVista_$SPLIT.json \
    --temperature 0 \
    --conv-mode $CONV_MODE

if [ "$SPLIT" = "testmini" ]; then
    python llava/eval/eval_mathvista.py --answer_file ./eval_output/$CKPT/MathVista/MathVista_$SPLIT.json
fi