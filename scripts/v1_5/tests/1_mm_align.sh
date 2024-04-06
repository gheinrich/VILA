#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
#export CUDA_LAUNCH_BLOCKING=1

###########################################################################
# cluster related information
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"
###########################################################################

export VISION_TOWER=${VISION_TOWER:-"google/siglip-large-patch16-384"}
VTOWER=$(echo $VISION_TOWER | rev | cut -d "/" -f 1 | rev)
# GLOBAL bs: 128 * 8
export ALIGN_DATASET=${ALIGN_DATASET:-llava_1_5_mm_align}


global_bs=${BATCH_SIZE:-128}
ACC_STEP=${ACC_STEP:-1}
bs=$((global_bs / n_node / ACC_STEP))
if [ "$n_node" = "1" ]; then
    #FIXME: set an extra to surprass the setting.
    echo "Detected on single machine. Automatically set batch size to 1 for debugging purpose."
    bs=1
fi
# bs=1 # for debug purpose

export BASE_MODEL_PATH=${BASE_MODEL_PATH:-"NousResearch/Llama-2-7b-hf"}
MNAME=$(echo $BASE_MODEL_PATH | rev | cut -d "/" -f 1 | rev)
OUTPUT_STEP1=${1:-"checkpoints/$MNAME-$VTOWER-align-$ALIGN_DATASET"}


echo "number of nodes:" $n_node
echo "per device batch size: $bs | global batch size $global_bs | base model: $BASE_MODEL_PATH"
echo "node rank:" $SLURM_PROCID
echo "ALIGN: $ALIGN_DATASET "


torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $BASE_MODEL_PATH \
    --version plain \
    --data_mixture $ALIGN_DATASET \
    --vision_tower $VISION_TOWER \
    --mm_projector mlp2x_gelu \
    --tune_mm_projector True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $OUTPUT_STEP1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACC_STEP \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --save_steps 10 \
    --save_total_limit 2 \
    --max_steps 30