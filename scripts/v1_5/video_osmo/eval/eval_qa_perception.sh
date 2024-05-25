#!/bin/bash
source ~/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python


cd ~/VILA


GPT_Zero_Shot_QA="./eval_output"
output_name=$1
output_dir="${GPT_Zero_Shot_QA}/${output_name}/PerceptionTest_Zero_Shot_QA"
pred_path="${GPT_Zero_Shot_QA}/${output_name}/PerceptionTest_Zero_Shot_QA/merge.jsonl"
output_json="${GPT_Zero_Shot_QA}/${output_name}/PerceptionTest_Zero_Shot_QA/results.json"
DATA_DIR="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/perception_test"
num_tasks=8


output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
if [ -f "$output_file" ]; then
    > "$output_file"
fi

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((num_tasks-1))); do
    cat ${output_dir}/${num_tasks}_${IDX}.json >> "$output_file"
done


python3 llava/eval/video/eval_video_perception.py \
    --pred_path ${pred_path} \
    --output_json ${output_json} \