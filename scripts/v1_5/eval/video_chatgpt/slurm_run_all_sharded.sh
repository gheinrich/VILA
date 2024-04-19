#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal
# Prerequisite: 1. "pip install openpyxl"; 2.Softlink "/home/yunhaof/workspace/datasets/evaluation" to "YOUR_VILA_PATH/playground/data/eval" before evaluation.

# Make sure partitions according to different clusters.
# PARTITIONS="batch_block1,batch_block2,batch_block3,batch_block4"
PARTITIONS="polar,grizzly"
ACCOUNT='llmservice_nlp_fm'
#ACCOUNT='nvr_elm_llm'

# Checkpoint path and model name (replace with your actual values)
checkpoint_path=$1
model_name=$2

mkdir -p eval_output/$model_name

srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_activitynet --gpus-per-node 8 --exclusive -o eval_output/$model_name/%J.activitynet.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_activitynet_sharded.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msvd --gpus-per-node 8 --exclusive -o eval_output/$model_name/%J.msvd.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msvd_sharded.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msrvtt --gpus-per-node 8 --exclusive -o eval_output/$model_name/%J.msrvtt.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msrvtt_sharded.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_tgif --gpus-per-node 8 --exclusive -o eval_output/$model_name/%J.tgif.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_tgif_sharded.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_nextqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/%J.nextqa.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_nextqa_sharded.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_perception --gpus-per-node 8 --exclusive -o eval_output/$model_name/%J.perception.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_perception_sharded.sh $checkpoint_path $model_name &
