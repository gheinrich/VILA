#!/bin/bash

source ~/.bashrc
conda activate vila_debug
which python

cd ~/workspace/VILA/

if [ ! -d "/tmp/mme" ]; then
    echo "Preparing dataset..."
    unzip -q /home/jil/workspace/LLaVA/playground/data/eval/MME/images.zip -d /tmp/mme
    echo "done"
else
    echo "Data already exists..."
fi

python -m llava.eval.eval_llava15_vqa \
    --model_name $1 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder /tmp/mme \
    --answers-file $1/eval/llava15/mme.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1_1

python playground/data/eval/MME/convert_answer_to_mme.py --experiment $1/eval/llava15/mme.jsonl

python playground/data/eval/MME/eval_tool/calculation.py --results_dir $1/eval/llava15/mme_results
