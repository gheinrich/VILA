#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/${USER}/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal

GPT_Zero_Shot_QA="eval/GPT_Zero_Shot_QA"
output_name=$1
pred_path="${GPT_Zero_Shot_QA}/NextQA_Zero_Shot_QA/${output_name}/merge.jsonl"
output_json="${GPT_Zero_Shot_QA}/NextQA_Zero_Shot_QA/${output_name}/results.json"
NEXTQA="/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/video_datasets_v2/nextqa"
stopwords_file="${NEXTQA}/stopwords.txt"
gt_file="${NEXTQA}/test_data_nextoe/test.csv"
num_tasks=8

cat ${GPT_Zero_Shot_QA}/NextQA_Zero_Shot_QA/${output_name}/${num_tasks}_*.json > ${pred_path}

python3 llava/eval/video/eval_video_nextqa.py \
    --pred_path ${pred_path} \
    --output_json ${output_json} \
    --gt_file ${gt_file} \
    --stopwords_file ${stopwords_file} \