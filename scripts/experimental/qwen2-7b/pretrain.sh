#!/bin/bash

DEFAULT_RUN_NAME="vila-qwen2-7b-pretrain"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=1024
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

source scripts/setups/train.sh

STAGE1_PATH=$1

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed ./scripts/zero3.json \
        --model_name_or_path $STAGE1_PATH \
        --version v1 \
        --data_mixture sharegpt4v_pretrain \
        --vision_tower google/siglip-so400m-patch14-384 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --tune_vision_tower False \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 200 \
        --save_total_limit 1 \
        --learning_rate 5e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 131072 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb
