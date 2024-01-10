#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/multi-modality-research/VILA/

python -m llava.eval.eval_mmmu \
    --output_path $1/eval/llava15/mmmu_output.json \
    --data_path ./playground/data/eval/mmmu/MMMU/ \
    --model_name $1 \
    --config_path ./playground/data/eval/mmmu/vila.yaml \
    --conv-mode vicuna_v1_1 \
    --split test