#!/bin/bash

# source ~/.bashrc
# conda activate vila
# which python
# cd ~/workspace/VILA-Internal
# Prerequisite:
#   1. "pip install openpyxl";
#   2.Softlink "/home/yunhaof/workspace/datasets/evaluation" to "YOUR_VILA_PATH/playground/data/eval" before evaluation.

# pip install openpyxl word2number mmengine openai
# ln -s /home/yunhaof/workspace/datasets/evaluation ./playground/data/eval

# Make sure partitions according to different clusters.
# PARTITIONS="batch_block1,batch_block2,batch_block3,batch_block4"
# llmservice_nlp_fm / nvr_elm_llm
SLURM_ACCOUNT=${SLURM_ACCOUNT:-"llmservice_nlp_fm"}
PARTITIONS=${SLURM_PARTITION:-"interactive,polar4,polar3,polar2,polar,batch_block1,grizzly,,batch_block2,batch_block3"}
echo "Submitting jobs to $PARTITIONS"

# Checkpoint path and model name (replace with your actual values)
checkpoint_path=$1
# model_name=$2
model_name=$(echo $checkpoint_path | rev | cut -d "/" -f 1 | rev)

echo $model_name
# Create output directory if it doesn't exist
mkdir -p eval_output/$model_name

## server evaluation benchmarks
# srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_vqav2 --gpus-per-node 8 --exclusive -o eval_output/$model_name/vqav2.txt ./scripts/v1_5/eval/vqav2.sh $checkpoint_path $model_name &
# srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_vizwiz --gpus-per-node 8 --exclusive -o eval_output/$model_name/vizwiz.txt ./scripts/v1_5/eval/vizwiz.sh $checkpoint_path $model_name &
# srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_mmbench --gpus-per-node 8 --exclusive -o eval_output/$model_name/mmbench.txt ./scripts/v1_5/eval/mmbench.sh $checkpoint_path $model_name &
# srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_mmbench_cn --gpus-per-node 8 --exclusive -o eval_output/$model_name/mmbench_cn.txt ./scripts/v1_5/eval/mmbench_cn.sh $checkpoint_path $model_name &
# srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_mmmu_test --gpus-per-node 8 --exclusive -o eval_output/$model_name/mmmu_test.txt ./scripts/v1_5/eval/mmmu.sh $checkpoint_path $model_name test &

## local evaluation benchmarks
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_mmmu_val --gpus-per-node 8 --exclusive -o eval_output/$model_name/mmmu_val.txt ./scripts/v1_5/eval/mmmu.sh $checkpoint_path $model_name validation &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_mathvista_testmini --gpus-per-node 8 --exclusive -o eval_output/$model_name/mathvista_testmini.txt ./scripts/v1_5/eval/mathvista.sh $checkpoint_path $model_name testmini &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_mathvista_test --gpus-per-node 8 --exclusive -o eval_output/$model_name/mathvista_test.txt ./scripts/v1_5/eval/mathvista.sh $checkpoint_path $model_name test &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_llavabench --gpus-per-node 8 --exclusive -o eval_output/$model_name/llavabench.txt ./scripts/v1_5/eval/llavabench.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_sqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/science.txt ./scripts/v1_5/eval/sqa.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_textvqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/textvqa.txt ./scripts/v1_5/eval/textvqa.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_mme --gpus-per-node 8 --exclusive -o eval_output/$model_name/mme.txt ./scripts/v1_5/eval/mme.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_mmvet --gpus-per-node 8 --exclusive -o eval_output/$model_name/mmvet.txt ./scripts/v1_5/eval/mmvet.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_pope --gpus-per-node 8 --exclusive -o eval_output/$model_name/pope.txt ./scripts/v1_5/eval/pope.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_seed --gpus-per-node 8 --exclusive -o eval_output/$model_name/seed.txt ./scripts/v1_5/eval/seed.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_gqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/gqa.txt ./scripts/v1_5/eval/gqa.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_chartqa --gpus-per-node 8 --exclusive -o eval_output/$model_name/chartqa.txt ./scripts/v1_5/eval/chartqa.sh $checkpoint_path $model_name &
srun -p $PARTITIONS -A $SLURM_ACCOUNT -N 1 -t 4:00:00 -J vila:eval_ai2d --gpus-per-node 8 --exclusive -o eval_output/$model_name/ai2d.txt ./scripts/v1_5/eval/ai2d.sh $checkpoint_path $model_name &
