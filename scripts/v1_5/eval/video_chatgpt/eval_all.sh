#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal

model_name=$1

echo activity_net
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_activitynet.sh ${model_name}
echo msvd
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_msvd.sh ${model_name}
echo msrvtt
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_msrvtt.sh ${model_name}
echo tgif
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_tgif.sh ${model_name}
echo nextqa
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_nextqa.sh ${model_name}
echo perception_test
bash ./scripts/v1_5/eval/video_chatgpt/eval_qa_perception.sh ${model_name}
echo video_mme
python llava/eval/video_mme/video_eval.py --output_dir=./eval_output/$model_name/video_mme/ --output_name=$model_name --convert
python llava/eval/video_mme/mme_calc.py --results_file ./eval_output/$model_name/video_mme/${model_name}_converted.json --video_duration_type short,medium,long --return_categories_accuracy --return_sub_categories_accuracy --return_task_types_accuracy