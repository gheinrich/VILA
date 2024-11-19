#!/bin/bash


DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=2048
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1
MM_VISION_SELECT_FEATURE="dense"
MM_VISION_SELECT_LAYER=-1
MM_PROJECTOR=mlp_downsample
VISION_TOWER="radio:768:nvidia/RADIO-L"
IMAGE_ASPECT_RATIO="resize"
MODEL="nvidia/Mistral-NeMo-Minitron-8B-Base"
OUTPUT_DIR=/tmp
DATASET="llava_1_5_mm_align"
CHAT_TEMPLATE="mistral"

REPORT_TO="wandb"

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
    --mm-vision-select-feature)
        MM_VISION_SELECT_FEATURE="$2"
        shift # past argument
        shift # past value
        ;;
    --mm-vision-select-layer)
        MM_VISION_SELECT_LAYER="$2"
        shift # past argument
        shift # past value
        ;;
    --vision-tower)
        VISION_TOWER="$2"
        shift # past argument
        shift # past value
        ;;
    --mm-projector)
        MM_PROJECTOR="$2"
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
    --model)
        MODEL="$2"
        shift # past argument
        shift # past value
        ;;
    --chat-template)
        CHAT_TEMPLATE="$2"
        shift # past argument
        shift # past value
        ;;
    --dataset)
        DATASET="$2"
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

#conda activate vila

# apt-get install slurm-client
conda run --no-capture-output  -n vila pip install hydra-core
conda run --no-capture-output  -n vila pip install loguru

export DEEPSPEED_LOG_LEVEL=DEBUG

export VILA_DATASETS=draco-oci-iad.yaml


# Overwrite Transformers
cp -r llava/train/transformers_replace/* /opt/conda/envs/vila/lib/python3.10/site-packages/transformers/

#NCCL_DEBUG=WARN SUBMIT_GPUS=1 PYTHONPATH=.:lib WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 conda run --no-capture-output  -n vila python -m debugpy --listen 0.0.0.0:3011 --wait-for-client \

conda run --no-capture-output  -n vila torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path ${MODEL} \
        --chat_template ${CHAT_TEMPLATE} \
        --data_mixture ${DATASET} \
        --vision_tower ${VISION_TOWER} \
        --mm_vision_select_feature ${MM_VISION_SELECT_FEATURE} \
        --mm_projector ${MM_PROJECTOR} \
        --tune_vision_tower False \
        --tune_mm_projector True \
        --tune_language_model False \
        --mm_vision_select_layer ${MM_VISION_SELECT_LAYER} \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio ${IMAGE_ASPECT_RATIO} \
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
        --report_to ${REPORT_TO}
