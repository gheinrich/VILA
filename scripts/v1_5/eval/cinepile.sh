#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

VIDEO_DIR="/home/xiuli/workspace/cinepile/yt_videos"
OUTPUT_DIR="runs/eval/$CKPT/cinepile"

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
GENERATION_CONFIG='{"max_new_tokens": 1024}'

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/cinepile.py \
    --model-path $MODEL_PATH \
    --conv-mode $CONV_MODE \
    --generation-config "$GENERATION_CONFIG" \
    --video-dir $VIDEO_DIR \
    --output-dir $OUTPUT_DIR
