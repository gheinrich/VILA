#!/bin/bash

DEFAULT_RUN_NAME="vila-llama2-7b-v1.5-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=2048
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

source scripts/setups/train.sh

STAGE2_PATH=$1

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE2_PATH \
        --version v1 \
        --data_mixture sharegpt4v_sft+vflan \
        --vision_tower google/siglip-so400m-patch14-384 \
        --mm_projector mlp_downsample \
        --tune_language_model True \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --mm_vision_select_layer -2 \
        --mm_vision_select_feature cls_patch \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --vflan_no_system_prompt True \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --lazy_preprocess True \
        --report_to wandb