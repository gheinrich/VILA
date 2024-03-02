#!/bin/bash

source ~/.bashrc
conda activate vila
which python
cd ~/workspace/VILA-Internal
# Prerequisite: 1. "pip install openpyxl"; 2.Softlink "/home/yunhaof/workspace/datasets/evaluation" to "YOUR_VILA_PATH/playground/data/eval" before evaluation.

# Make sure partitions according to different clusters.
PARTITIONS="batch_block1,batch_block2,batch_block3,batch_block4"
# PARTITIONS="batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly"

# Checkpoint path and model name (replace with your actual values)
checkpoint_path=$1
model_name=$2

# Create output directory if it doesn't exist
mkdir -p eval_output/$model_name

## server evaluation benchmarks
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_vqav2 --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.vqav2.txt ./scripts/v1_5/eval/vqav2.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_vizwiz --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.vizwiz.txt ./scripts/v1_5/eval/vizwiz.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_mmbench --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.mmbench.txt ./scripts/v1_5/eval/mmbench.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_mmbench_cn --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.mmbench_cn.txt ./scripts/v1_5/eval/mmbench_cn.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_mmmu_test --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.mmmu_test.txt ./scripts/v1_5/eval/mmmu.sh $checkpoint_path $model_name test &
## local evaluation benchmarks
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_mmmu_val --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.mmmu_val.txt ./scripts/v1_5/eval/mmmu.sh $checkpoint_path $model_name validation &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_llavabench --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.llavabench.txt ./scripts/v1_5/eval/llavabench.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_sqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.science.txt ./scripts/v1_5/eval/sqa.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_textvqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.textvqa.txt ./scripts/v1_5/eval/textvqa.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_mme --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.mme.txt ./scripts/v1_5/eval/mme.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_mmvet --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.mmvet.txt ./scripts/v1_5/eval/mmvet.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_pope --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.pope.txt ./scripts/v1_5/eval/pope.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_seed --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.seed.txt ./scripts/v1_5/eval/seed.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm:evaluation_gqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/%N.gqa.txt ./scripts/v1_5/eval/gqa.sh $checkpoint_path $model_name &
