#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal

model_name=$1

echo msvd
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_msvd.sh ${model_name}
echo msrvtt
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_msrvtt.sh ${model_name}
echo tgif
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_tgif.sh ${model_name}

echo vila_benchmark
echo pexels
output_dir="runs/eval/${model_name}/VILA-benchmark/pexels"
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} pexels correctness ${output_dir}/pred.json
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} pexels detailed_orientation ${output_dir}/pred.json
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} pexels context ${output_dir}/pred.json

output_dir="runs/eval/${model_name}/VILA-benchmark/robotics"
echo robotics
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} robotics correctness ${output_dir}/pred.json
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} robotics detailed_orientation ${output_dir}/pred.json
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} robotics context ${output_dir}/pred.json

output_dir="runs/eval/${model_name}/VILA-benchmark/av"
echo av

bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} av correctness ${output_dir}/pred.json
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} av detailed_orientation ${output_dir}/pred.json
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} av context ${output_dir}/pred.json

output_dir="runs/eval/${model_name}/VILA-benchmark/long"
echo cartoon
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} cartoon correctness ${output_dir}/pred.json
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} cartoon detailed_orientation ${output_dir}/pred.json
bash ./scripts/v1_5/eval/video_chatgpt/eval_vila_benchmark_gpt4.sh ${model_name} cartoon context ${output_dir}/pred.json
