#!/bin/bash

model_path=$1
CKPT_NAME=$2
GPT_Zero_Shot_QA="~/workspace/VILA-Internal/eval/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/all_test"
gt_file_question="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/test_a.json"
output_dir="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${CKPT_NAME}"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=$(( ${#GPULIST[@]} / 2 )) # Calculate chunks for 2 GPUs per chunk


for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_IDX1=$((IDX * 2))  # First GPU index
    GPU_IDX2=$((GPU_IDX1 + 1))  # Second GPU index

    CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX1]},${GPULIST[$GPU_IDX2]} python3 llava/eval/model_vqa_video.py \
      --model-path ${model_path} \
      --video_dir ${video_dir} \
      --model_max_length 8192 \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num-chunks $CHUNKS \
      --chunk-idx $IDX \
      --conv-mode hermes-2 \
      --temperature 0 &
done

wait

output_file=${output_dir}/merge.jsonl

> "$output_file"

CHUNKS=4

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> $output_file
done