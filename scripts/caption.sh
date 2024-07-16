#!/bin/bash
source ~/.bashrc
source activate vila
which python

cd ~/workspace/multi-modality-research/VILA/

IDX=${1:-1}
TOTAL=${2:-1}
########  slurm related env vars
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}

worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
echo "JobID: $SLURM_JOB_ID Full worker list: $worker_list"
echo "MASTER_ADDR="$MASTER_ADDR
n_node=${SLURM_JOB_NUM_NODES:-1}

########  running jobs
# torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
#     --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
#     llava/eval/run_llava_list_coyo.py \
#     --model-name ~/downloads/vicuna-13b-clip336-mmc4sub+coyo-finetune-captioner-e4 \
#     --conv-mode vicuna_v1_1

torchrun  --nproc_per_node=8  llava/eval/run_llava_list_coyo.py \
    --model-name ~/downloads/vicuna-13b-clip336-mmc4sub+coyo-finetune-captioner-e4 \
    --conv-mode vicuna_v1_1 \
    --idx $IDX \
    --total $TOTAL

# torchrun  --nproc_per_node=8  llava/eval/run_llava_list_coyo.py \
#     --model-name ~/downloads/vicuna-13b-clip336-mmc4sub+coyo-finetune-captioner-e4 \
#     --conv-mode vicuna_v1_1 \
#     --idx 1 \
#     --total 4


# python -W ignore llava/eval/run_llava_list.py \
#     --model-name $1 \
#     --conv-mode vicuna_v1_1 \
#     --dataset $2 \
#     --idx $3 \
#     --total $4
