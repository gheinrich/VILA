#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 4 ]; then
    CONV_MODE="$4"
fi

EGOSCHEMA_PATH=/lustre/fsw/portfolios/nvr/users/xiuli/EgoSchema

CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_vqa_ego_schema \
    --model-path $MODEL_PATH \
    --temperature 0 \
    --video-folder $EGOSCHEMA_PATH/videos \
    --question-file $EGOSCHEMA_PATH/questions.json \
    --gt-answers-file $EGOSCHEMA_PATH/subset_answers.json \
    --conv-mode $CONV_MODE \
    --output_path ./eval_output/$CKPT/EgoSchema/answers.json \