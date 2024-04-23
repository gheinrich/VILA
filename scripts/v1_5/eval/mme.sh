#!/bin/bash
MODEL_PATH=$1
CKPT=$2
MMEDIR="./playground/data/eval/MME"

mkdir -p ./playground/data/eval/MME/answers/$CKPT

# TODO(yunhao,ligeng): change the following to the correct device
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./eval_output/$CKPT/MME/mme.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python $MMEDIR/convert_answer_to_mme.py --experiment ./eval_output/$CKPT/MME/mme.jsonl

python $MMEDIR/eval_tool/calculation.py --results_dir ./eval_output/$CKPT/MME/mme_results

