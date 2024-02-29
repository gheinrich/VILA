#!/bin/bash

# Prerequisite: Softlink "/home/yunhaof/workspace/datasets/evaluation" to "YOUR_VILA_PATH/playground/data/eval" before evaluation.

# Checkpoint path and model name (replace with your actual values)
checkpoint_path=$1
model_name=$2

# Create output directory if it doesn't exist
mkdir -p eval_output/$model_name

## server evaluation benchmarks
srun -p batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly -A nvr_elm_llm -N 1 -t 4:00:00 -J evaluation_vizwiz --gpus-per-node 8 --exclusive -o eval_output/$model_name/$SLURM_JOB_ID_%N.vizwiz.txt ./scripts/v1_5/eval/vizwiz.sh $checkpoint_path $model_name &
srun -p batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly -A nvr_elm_llm -N 1 -t 4:00:00 -J evaluation_mmbench --gpus-per-node 8 --exclusive -o eval_output/$model_name/$SLURM_JOB_ID_%N.mmbench.txt ./scripts/v1_5/eval/mmbench.sh $checkpoint_path $model_name &
srun -p batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly -A nvr_elm_llm -N 1 -t 4:00:00 -J evaluation_mmmu --gpus-per-node 8 --exclusive -o eval_output/$model_name/$SLURM_JOB_ID_%N.mmmu.txt ./scripts/v1_5/eval/mmmu.sh $checkpoint_path $model_name &
## local evaluation benchmarks
srun -p batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly -A nvr_elm_llm -N 1 -t 4:00:00 -J evaluation_sqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/$SLURM_JOB_ID_%N.science.txt ./scripts/v1_5/eval/sqa.sh $checkpoint_path $model_name &
srun -p batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly -A nvr_elm_llm -N 1 -t 4:00:00 -J evaluation_textvqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/$SLURM_JOB_ID_%N.textvqa.txt ./scripts/v1_5/eval/textvqa.sh $checkpoint_path $model_name &
srun -p batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly -A nvr_elm_llm -N 1 -t 4:00:00 -J evaluation_mme --gpus-per-node 8 --exclusive -o eval_output/$model_name/$SLURM_JOB_ID_%N.mme.txt ./scripts/v1_5/eval/mme.sh $checkpoint_path $model_name &
srun -p batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly -A nvr_elm_llm -N 1 -t 4:00:00 -J evaluation_pope --gpus-per-node 8 --exclusive -o eval_output/$model_name/$SLURM_JOB_ID_%N.pope.txt ./scripts/v1_5/eval/pope.sh $checkpoint_path $model_name &
srun -p batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly -A nvr_elm_llm -N 1 -t 4:00:00 -J evaluation_seed --gpus-per-node 8 --exclusive -o eval_output/$model_name/$SLURM_JOB_ID_%N.seed.txt ./scripts/v1_5/eval/seed.sh $checkpoint_path $model_name &
## long-time evaluation benchmarks
srun -p batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly -A nvr_elm_llm -N 1 -t 4:00:00 -J evaluation_vqav2 --gpus-per-node 8 --exclusive -o eval_output/$model_name/$SLURM_JOB_ID_%N.vqav2.txt ./scripts/v1_5/eval/vqav2.sh $checkpoint_path $model_name &
srun -p batch_block1,batch_block2,batch_block3,batch_block4,polar,grizzly -A nvr_elm_llm -N 1 -t 4:00:00 -J evaluation_gqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/$SLURM_JOB_ID_%N.gqa.txt ./scripts/v1_5/eval/gqa.sh $checkpoint_path $model_name &
