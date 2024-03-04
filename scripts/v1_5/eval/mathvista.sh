#!/bin/bash
MODEL_PATH=$1
CKPT=$2
SPLIT=$3

mkdir -p ./playground/data/eval/MathVista/answers/$CKPT

python -m llava.eval.eval_mathvista \
    --model-path $MODEL_PATH \
    --split $SPLIT \
    --answers-file ./playground/data/eval/MathVista/answers/$CKPT/MathVista_$SPLIT.json \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MathVista

python convert_answer_to_mme.py --experiment answers/$CKPT/mme.jsonl

cd eval_tool

python calculation.py --results_dir ../answers/$CKPT/mme_results

