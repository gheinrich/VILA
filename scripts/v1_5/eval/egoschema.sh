#!/bin/bash

SPLIT=$1
MODEL_PATH=$2
CONV_MODE=$3

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/egoschema_$SPLIT"}

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 1024}'

DATA_PATH="/home/yunhaof/workspace/datasets/evaluation/EgoSchema/questions.json"
VIDEO_DIR="/home/yunhaof/workspace/datasets/evaluation/EgoSchema/videos"
ANSWER_PATH="/home/yunhaof/workspace/datasets/evaluation/EgoSchema/subset_answers.json"

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/egoschema.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --data-path $DATA_PATH \
    --video-dir $VIDEO_DIR \
    --split $SPLIT \
    --answer-path $ANSWER_PATH \
    --output-dir $OUTPUT_DIR
