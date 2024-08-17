#!/bin/bash

DEFAULT_RUN_NAME="vila1.5-8b-lita-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=128
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

source scripts/setups/train.sh

DATA_MIXTURE=(
    "llava-instruct-150k"
    "sharegpt4v-instruct-100k"
    "activitynet-dvc*3"
    "youcook2-dvc*3"
    "medical-dvc"
    "warehouse-dvc"
    "activitynet-el*3"
    "youcook2-el*3"
    "didemo-el"
    "charades-el"
    "medical-el"
    "warehouse-el"
    "nextqa"
    "activitynet-rtl"
)
IFS=$'\n' DATA_MIXTURE=($(sort <<<"${DATA_MIXTURE[*]}"))
DATA_MIXTURE=$(IFS=+; echo "${DATA_MIXTURE[*]}")
echo "DATA_MIXTURE = $DATA_MIXTURE"

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path Efficient-Large-Model/Llama-3-VILA1.5-8B-Fix \
        --version llama_3 \
        --data_mixture $DATA_MIXTURE \
        --vision_tower google/siglip-so400m-patch14-384 \
        --mm_projector mlp_downsample \
        --tune_language_model True \
        --tune_vision_tower False \
        --tune_mm_projector False \
        --mm_vision_select_layer -2 \
        --mm_vision_select_feature cls_patch \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --num_video_frames 16 \
        --num_time_tokens 100 \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb
