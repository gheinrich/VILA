#!/bin/bash
source ~/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python


cd ~/VILA


GPT_Zero_Shot_QA="./eval_output"
output_name=$1
if [ -z "$2" ]
then
    gpt_model="gpt-3.5-turbo"
else
    gpt_model=$2
fi
pred_path="${GPT_Zero_Shot_QA}/${output_name}/Activitynet_Zero_Shot_QA/merge.jsonl"
output_dir="${GPT_Zero_Shot_QA}/${output_name}/Activitynet_Zero_Shot_QA/${gpt_model}"
output_json="${GPT_Zero_Shot_QA}/${output_name}/Activitynet_Zero_Shot_QA/results.json"
num_tasks=8

echo "Your API key starts with: ${OPENAI_API_KEY:0:5}"

python3 llava/eval/video/eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --num_tasks ${num_tasks} \
    --gpt_model ${gpt_model}