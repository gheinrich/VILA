#!/bin/bash
MODEL_PATH=$1
CKPT=$2
SPLIT=$3


CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_mathvista \
    --model-path $MODEL_PATH \
    --split $SPLIT \
    --answers-file ./eval_output/$CKPT/MathVista/MathVista_$SPLIT.json \
    --temperature 0 \
    --conv-mode vicuna_v1

if [ "$SPLIT" = "testmini" ]; then
    python llava/eval/eval_mathvista.py --answer_file ./eval_output/$CKPT/MathVista/MathVista_$SPLIT.json
fi