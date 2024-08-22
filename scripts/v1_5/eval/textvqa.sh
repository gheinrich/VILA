#!/bin/bash

SPLIT=$1
MODEL_PATH=$2
CONV_MODE=$3

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/textvqa"}

DATA_PATH="/home/yunhaof/workspace/datasets/evaluation/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_DIR="/home/yunhaof/workspace/datasets/evaluation/textvqa/train_images"
ANSWER_PATH="/home/yunhaof/workspace/datasets/evaluation/textvqa/TextVQA_0.5.1_val.json"

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 128}'

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/textvqa.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --data-path $DATA_PATH \
    --image-dir $IMAGE_DIR \
    --answer-path $ANSWER_PATH \
    --output-dir $OUTPUT_DIR
