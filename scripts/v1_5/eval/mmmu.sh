#!/bin/bash

SPLIT=$1
MODEL_PATH=$2
CONV_MODE=$3

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR=${OUTPUT_DIR:-"runs/eval/$MODEL_NAME/mmmu_$SPLIT"}

DATA_PATH="/home/yunhaof/workspace/datasets/evaluation/MMMU"
ANSWER_PATH="/home/yunhaof/workspace/datasets/evaluation/MMMU/answer_dict_val.json"

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 128, "do_sample": true, "num_beams": 5}'

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/mmmu.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --data-path $DATA_PATH \
    --split $SPLIT \
    --answer-path $ANSWER_PATH \
    --output-dir $OUTPUT_DIR
