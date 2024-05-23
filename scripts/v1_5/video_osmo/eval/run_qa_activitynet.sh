#!/bin/bash
source ~/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python


cd ~/VILA


model_path=$1
CKPT_NAME=$2
CONV_MODE=llama_3
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi
GPT_Zero_Shot_QA="/mnt/amlfs-01/home/fuzhaox/video_datasets_v2/GPT_Zero_Shot_QA/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/all_test"
gt_file_question="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/test_a.json"
output_dir="./eval_output/${CKPT_NAME}/Activitynet_Zero_Shot_QA"

CHUNKS=8


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=$IDX python3 llava/eval/model_vqa_video.py \
      --model-path ${model_path} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num-chunks $CHUNKS \
      --chunk-idx $IDX \
      --conv-mode $CONV_MODE \
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