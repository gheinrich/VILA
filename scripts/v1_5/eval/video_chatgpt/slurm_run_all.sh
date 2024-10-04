#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal
# Prerequisite: 1. pip install -e ".[eval]";

# Make sure partitions according to different clusters.
#PARTITIONS="batch_block1,batch_block2,batch_block3,batch_block4"
PARTITIONS="polar,grizzly"
#ACCOUNT='llmservice_nlp_fm'
ACCOUNT='nvr_elm_llm'

# Checkpoint path and model name (replace with your actual values)
checkpoint_path=$1
model_name=$2
conv_mode=vicuna_v1
if [ "$#" -ge 3 ]; then
    conv_mode="$3"
fi

mkdir -p runs/eval/$model_name

srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msvd --gpus-per-node 8 --exclusive -o runs/eval/$model_name/%J.msvd.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msvd.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msrvtt --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.msrvtt.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msrvtt.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_msrvtt --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.msrvtt.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_msrvtt.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_tgif --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.tgif.txt ./scripts/v1_5/eval/video_chatgpt/run_qa_tgif.sh $checkpoint_path $model_name $conv_mode &
srun -p $PARTITIONS -A $ACCOUNT -N 1 -t 4:00:00 -J $ACCOUNT:evaluation_vila_benchmark --gpus-per-node 8 --dependency singleton --exclusive -o runs/eval/$model_name/%J.vila_benchmark.txt ./scripts/v1_5/eval/video_chatgpt/run_vila_benchmark.sh $checkpoint_path $model_name $conv_mode &
