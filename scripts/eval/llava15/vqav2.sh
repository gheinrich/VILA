#!/bin/bash

source ~/.bashrc
conda activate vila_debug
which python

if [ ! -d "/tmp/vqav2" ]; then
    echo "Preparing dataset..."
    unzip -q /home/jil/workspace/LLaVA/playground/data/eval/vqav2/test2015.zip -d /tmp/vqav2
    echo "done"
else
    echo "Data already exists..."
fi

cd ~/workspace/VILA/



gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=$1
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_llava15_vqa \
        --model_name $1 \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder /tmp/vqav2/test2015/ \
        --answers-file $1/eval/llava15/vqav2/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1_1 &
done

wait

output_file=$1/eval/llava15/vqav2/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $1/eval/llava15/vqav2/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --ckpt $CKPT
