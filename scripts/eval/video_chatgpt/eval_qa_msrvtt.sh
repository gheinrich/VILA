#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/multi-modality-research/VILA/

GPT_Zero_Shot_QA="eval/GPT_Zero_Shot_QA"
output_name=$1
pred_path="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${output_name}/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${output_name}/gpt"
output_json="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${output_name}/results.json"
num_tasks=8



python3 llava/eval/video/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --num_tasks ${num_tasks}
