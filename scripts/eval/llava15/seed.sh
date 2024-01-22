#!/bin/bash

source ~/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python



cd ~/workspace/VILA/

if [ ! -d "/tmp/seed" ]; then
    echo "Preparing dataset..."
    unzip -q /home/jil/workspace/LLaVA/playground/data/eval/seed_bench/SEED-Bench-image.zip -d /tmp/seed
    unzip -q /home/jil/workspace/LLaVA/playground/data/eval/seed_bench/SEED-Bench-video-image.zip -d /tmp/seed
    echo "done"
else
    echo "Data already exists..."
fi


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

echo $CHUNKS 

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.eval_llava15_vqa \
        --model_name $1 \
        --question-file ./playground/data/eval/seed_bench/llava-seed-bench.jsonl \
        --image-folder /tmp/seed/ \
        --answers-file $1/eval/llava15/seed/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1_1 &
done

wait

output_file=$1/eval/llava15/seed/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $1/eval/llava15/seed/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file output_file=$1/eval/llava15/seed/upload.jsonl
