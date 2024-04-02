#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/${USER}/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal

model_path=$1
CKPT_NAME=$2
PROMPT_TEMPLATE=$3
GPT_Zero_Shot_QA="~/workspace/VILA-Internal/eval/GPT_Zero_Shot_QA"
DEMO_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/vila-captioner-avfm"
video_dir="${DEMO_DIR}/videos"
# gt_file="${NEXTQA}/test_data_nextoe/test.csv"
output_dir="${GPT_Zero_Shot_QA}/Demo_Zero_Shot_QA/${CKPT_NAME}_${PROMPT_TEMPLATE}"
# mkdir $output_dir

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}


# for IDX in $(seq 0 $((CHUNKS-1))); do
# CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} 
python3 llava/eval/model_vqa_videodemo.py \
    --model-path ${model_path} \
    --video_dir ${video_dir} \
    --model_max_length 8192 \
    --output_dir ${output_dir} \
    --prompt_template ${PROMPT_TEMPLATE} \
    --conv-mode vicuna_v1 \
    --temperature 0 &
# done

# wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
if [ -f "$output_file" ]; then
    > "$output_file"
fi

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done