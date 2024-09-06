#!/bin/bash

DEFAULT_RUN_NAME="vila-mistral-nemo-minitron-8b-align"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=2048
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

source scripts/setups/train.sh

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path nvidia/Mistral-NeMo-Minitron-8B-Base \
        --chat_template mistral \
        --data_mixture llava_1_5_mm_align \
        --vision_tower google/siglip-so400m-patch14-384 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --tune_vision_tower False \
        --tune_mm_projector True \
        --tune_language_model False \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio resize \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --report_to wandb
