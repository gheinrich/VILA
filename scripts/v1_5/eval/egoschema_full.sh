#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

OUTPUT_DIR="runs/eval/$CKPT/egoschema-full"

NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}

torchrun --nproc-per-node=$NPROC_PER_NODE \
    llava/eval/model_vqa_ego_schema.py \
    --model-path $MODEL_PATH \
    --generation-config '{"max_new_tokens": 1024}' \
    --video-folder playground/data/eval/EgoSchema/videos \
    --question-file playground/data/eval/EgoSchema/questions.json \
    --gt-answers-file playground/data/eval/EgoSchema/subset_answers.json \
    --conv-mode $CONV_MODE \
    --split test \
    --output_dir $OUTPUT_DIR \
    --output_name merge \

# convert json to csv for kaggle submission
python scripts/v1_5/eval/convert_pred_to_csv.py --output_dir $OUTPUT_DIR
