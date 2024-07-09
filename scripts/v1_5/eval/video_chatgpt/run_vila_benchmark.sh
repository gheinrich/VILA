#!/bin/bash

model_path=$1
model_name=$2
conv_mode=$3

echo vila_benchmark
output_dir="./eval_output/VILA-benchmark/${model_name}"
python llava/eval/video/model_vqa_videodemo_benchmark.py --model-path ${model_path} --eval_type pexels --output_dir ${output_dir} --conv-mode ${conv_mode} --temperature 0
python llava/eval/video/model_vqa_videodemo_benchmark.py --model-path ${model_path} --eval_type robotics --output_dir ${output_dir} --conv-mode ${conv_mode} --temperature 0
python llava/eval/video/model_vqa_videodemo_benchmark.py --model-path ${model_path} --eval_type av --output_dir ${output_dir} --conv-mode ${conv_mode} --temperature 0
