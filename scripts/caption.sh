#!/bin/bash
source ~/.bashrc
source activate vila
which python

cd ~/workspace/multi-modality-research/VILA/

python -W ignore llava/eval/run_llava_list.py \
    --model-name $1 \
    --conv-mode vicuna_v1_1 \
    --dataset $2 \
    --idx $3 \
    --total $4