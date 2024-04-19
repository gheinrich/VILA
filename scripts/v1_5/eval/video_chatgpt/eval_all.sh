#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal

model_name=$2

bash /home/jasonlu/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_activitynet.sh $model_name > activitynet.txt &
bash /home/jasonlu/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_msvd.sh $model_name > msvd.txt &
bash /home/jasonlu/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_msrvtt.sh $model_name > msrvtt.txt &
bash /home/jasonlu/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_tgif.sh $model_name > tgif.txt &
bash /home/jasonlu/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_nextqa.sh $model_name > nextqa.txt &
bash /home/jasonlu/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_perception.sh $model_name > perception.txt &
