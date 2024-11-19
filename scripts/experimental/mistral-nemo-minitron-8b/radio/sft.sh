#!/bin/bash

DEFAULT_RUN_NAME="vila-mistral-nemo-minitron-8b-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=2048
DEFAULT_GRADIENT_ACCUMULATION_STEPS=16

OUTPUT_DIR=/tmp
STAGE2_PATH=/specify/your/path/to/stage2

REPORT_TO="wandb"

DATASET="sharegpt4v_sft+vflan"

IMAGE_ASPECT_RATIO="resize"

LEARNING_RATE=1e-4

TUNE_VISION_TOWER=False

MODEL_MAX_LENGTH=4096

export VILA_DATASETS=draco-oci-iad.yaml

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output)
      OUTPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -r|--report-to)
        REPORT_TO="$2"
        shift # past argument
        shift # past value
        ;;
    --batch-size)
        DEFAULT_GLOBAL_TRAIN_BATCH_SIZE="$2"
        shift # past argument
        shift # past value
        ;;
    --stage2-path)
        STAGE2_PATH="$2"
        shift # past argument
        shift # past value
        ;;
    --dataset)
        DATASET="$2"
        shift # past argument
        shift # past value
        ;;
    --gradient-accumulation-steps)
        DEFAULT_GRADIENT_ACCUMULATION_STEPS="$2"
        shift # past argument
        shift # past value
        ;;
    --image-aspect-ratio)
        IMAGE_ASPECT_RATIO="$2"
        shift # past argument
        shift # past value
        ;;
    --learning-rate)
        LEARNING_RATE="$2"
        shift # past argument
        shift # past value
        ;;
    --tune-vision-tower)
        TUNE_VISION_TOWER="$2"
        shift # past argument
        shift # past value
        ;;
    --model-max-length)
        MODEL_MAX_LENGTH="$2"
        shift # past argument
        shift # past value
        ;;
    *)
      shift # past argument
      ;;
  esac
done

DEFAULT_RUN_NAME=$(basename $OUTPUT_DIR)


source ~/.bashrc
source scripts/setups/train.sh

# Overwrite Transformers
cp -r llava/train/transformers_replace/* /opt/conda/envs/vila/lib/python3.10/site-packages/transformers/

#conda activate vila

conda run --no-capture-output  -n vila pip install hydra-core
conda run --no-capture-output  -n vila pip install loguru

conda run --no-capture-output  -n vila torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE2_PATH \
        --data_mixture ${DATASET} \
        --mm_projector mlp_downsample \
        --tune_vision_tower ${TUNE_VISION_TOWER} \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio ${IMAGE_ASPECT_RATIO} \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --save_strategy steps \
        --save_steps 50 \
        --save_total_limit 1 \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length ${MODEL_MAX_LENGTH} \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to ${REPORT_TO}
