#!/bin/bash

DEFAULT_RUN_NAME="vila-40b-neurips-pretrain"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=1024
DEFAULT_GRADIENT_ACCUMULATION_STEPS=2

source scripts/setups/train.sh

OUTPUT_DIR=${OUTPUT_DIR:-"runs/dev"}

if [ "$NNODES" = "1" ] || [ "$NNODES" = "2" ]; then
    echo "Detected on single machine. Automatically set batch size to 1 for debugging purpose."
    PER_DEVICE_TRAIN_BATCH_SIZE=1
fi

# BASE_MODEL="runs/train/1_alignment-nodes_16/model"
# DATASETS="mmc4core_10_subset+coyo25m_0to10_vila15_40b_recap+sharegpt4v_pretrain"
DATASETS="sharegpt4v_pretrain+mmc4core_10_subset+coyo_25m_wds_spatial_ocr_bbox_interleaved_qas"

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /home/jasonlu/models/Nous-Hermes-2-Yi-34B \
    --version hermes-2 \
    --data_mixture $DATASETS \
    --vision_tower /home/jasonlu/models/InternViT-6B-448px-V1-2 \
    --mlp_path /home/jasonlu/models/InternViT-6B-448px-V1-2/mlp_projector.pth \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR/model \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb
