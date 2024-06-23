#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi
output_dir="./eval_output/$CKPT/EgoSchema"

echo "Evaluating $CKPT with conv_mode $CONV_MODE..."
CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_vqa_ego_schema \
    --model-path $MODEL_PATH \
    --temperature 0 \
    --conv-mode $CONV_MODE \
    --output_dir ${output_dir} \
    --output_name answers \
    --split validation \