#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/${USER}/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal

model_path=$1
CKPT_NAME=$2
GPT_Zero_Shot_QA="~/workspace/VILA-Internal/eval/GPT_Zero_Shot_QA"
NEXTQA="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/nextqa"
video_dir="${NEXTQA}/NExTVideo"
gt_file="${NEXTQA}/test_data_nextoe/test.csv"
output_dir="${GPT_Zero_Shot_QA}/NextQA_Zero_Shot_QA/${CKPT_NAME}"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 llava/eval/model_vqa_nextqa.py \
      --model-path ${model_path} \
      --video_dir ${video_dir} \
      --model_max_length 8192 \
      --gt_file ${gt_file} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num-chunks $CHUNKS \
      --chunk-idx $IDX \
      --conv-mode vicuna_v1 \
      --temperature 0 &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
if [ -f "$output_file" ]; then
    > "$output_file"
fi

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done