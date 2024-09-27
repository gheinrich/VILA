#!/bin/bash

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

EXTENDED_512k_PATH=$1
OUTPUT=$2
DATA_FILE=$3 # /lustre/fs2/portfolios/nvr/users/yukangc/datasets/SlimPajama-encode-llama3/encode_words_llama3_1M_first10k.jsonl

model_max_length=1048576
rope_theta=3580165449

mkdir -p $OUTPUT

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
        --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
        llava/train/train_llm_to_long.py  \
        --model_name_or_path $EXTENDED_512k_PATH \
        --bf16 True \
        --data_file $DATA_FILE \
        --output_dir $OUTPUT       \
        --cache_dir ./cache-1M \
        --model_max_length $model_max_length \
        --data_max_length $model_max_length \
        --rope_theta $rope_theta \
        --use_flash_attn True \
        --low_rank_training True \
        --max_steps 50  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 4     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 10     \
        --save_total_limit 2     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 2     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --deepspeed "./scripts/zero3.json" \
        --tf32 True
