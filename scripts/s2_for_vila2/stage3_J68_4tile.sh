#!/bin/bash

DEFAULT_RUN_NAME="stage3_J68_4tile"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=1024
DEFAULT_GRADIENT_ACCUMULATION_STEPS=4
STAGE2_PATH="runs/train/stage15_J68_4tile/model"

source scripts/setups/train.sh

export NCCL_IB_TIMEOUT=31

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE2_PATH \
        --data_mixture sharegpt4v_gpt4_100k+llava_instruct+sharegpt4v_sft+dvqa_train_200k+chartqa_train_18k+ai2d_train_12k+docvqa_train_10k+geoqa+synthdog_en \
        --vision_tower /home/jasonlu/models/paligemma-siglip-so400m-patch14-448 \
        --s2 True \
        --s2_scales "448,896,1344" \
        --s2_max_split_size 448 \
        --s2_resize_output_to_scale_idx 1 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample \
        --tune_vision_tower True \
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
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 20 \
        --save_total_limit 1 \
        --learning_rate 5e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 8192 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to wandb
