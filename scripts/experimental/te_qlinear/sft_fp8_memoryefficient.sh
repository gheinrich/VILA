#!/bin/bash

DEFAULT_RUN_NAME="vila-llama3-8b-s2-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=1024
DEFAULT_GRADIENT_ACCUMULATION_STEPS=2

source scripts/setups/train.sh

STAGE2_PATH=${1:-"/home/ligengz/workspace/checkpoints/yunhaof-july5-llama3_8b-mmc4_spatial_ocr_coyo-stage2"}
OUTPUT_DIR=${OUTPUT_DIR:-"runs/dev"}
STAGE3_DATASET=${STAGE3_DATASET:-"sharegpt4v_sft"}

if [ "$NNODES" = "1" ] || [ "$NNODES" = "2" ]; then
    echo "Detected on single machine. Automatically set batch size to 1 for debugging purpose."
    PER_DEVICE_TRAIN_BATCH_SIZE=1
fi

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE2_PATH \
        --version llama_3 \
        --data_mixture $STAGE3_DATASET \
        --vision_tower google/siglip-so400m-patch14-384 \
        --s2 True \
        --s2_scales "384,768" \
        --s2_max_split_size 384 \
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
        --save_steps 50 \
        --save_total_limit 1 \
        --learning_rate 1e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to wandb \
        --quantize_model "qmem" \
        --qchoice all \
        --fabit E4M3 \
        --fwbit E4M3 \
        --fobit E4M3 \
        --bwbit E5M2 \
        --babit E5M2 \
        --bobit E5M2 \
        --row_blocksize -1 \
        --col_blocksize -1 \
        --min_blockunit_row 1 \
        --min_blockunit_col 16 \
        --refine_residual_fp true \
        --refine_attn_blocksize true \
        --refine_mlp_blocksize true \
        --refine_row_blocksize 1 \
        --refine_col_blocksize 16 \
        --pad_to_multiple_of 4
