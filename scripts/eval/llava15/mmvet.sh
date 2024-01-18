#!/bin/bash
source ~/.bashrc
conda activate vila_debug
which python

if [ ! -d "/tmp/mmvet" ]; then
    echo "Preparing dataset..."
    unzip -q /home/jil/workspace/LLaVA/playground/data/eval/mm-vet/mm-vet.zip -d /tmp/mmvet
    echo "done"
else
    echo "Data already exists..."
fi

cd ~/workspace/VILA/

python -m llava.eval.model_vqa \
    --model-name $1 \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /tmp/mmvet/mm-vet//images \
    --answers-file $1/eval/llava15/mmvet.jsonl  \
    --temperature 0 \
    --conv-mode vicuna_v1_1

python scripts/convert_mmvet_for_eval.py \
    --src $1/eval/llava15/mmvet.jsonl \
    --dst $1/eval/llava15/mmvet-convert.jsonl


python llava/eval/eval_mmvet.py --results_file $1/eval/llava15/mmvet-convert.jsonl
