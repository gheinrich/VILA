#!/bin/bash

: '
Sample invocations

# Commercial Training on ORD
./run_job.py     --depchain 2  --image /lustre/fsw/portfolios/llmservice/users/gheinrich/cache/vila-7591ca.sqsh/vila-7591ca.sqsh \
  --nodes 8 --output /lustre/fsw/portfolios/nvr/users/gheinrich/results/23-11-24-vila-qwen2-cial-tile  \
  ./scripts/experimental/radio/train-all.sh  \
  --chat-template qwen2 \
  --vision-tower  radio:768:nvidia/C-RADIO \
  --stage1-dataset nvclip_5m_vfc_recap30 \
  --stage2-dataset nvclip_5m_vfc_recap70 \
  --stage3-dataset nvclip_conversation_complex_filt+nvclip_conversation_complex_new_unfilt+nvclip_ocr+oi_vila_caption+ln_caption_vila_augment+nv_metropolis_textcaps+textcaps_commercial+textvqa_commercial+nv_metropolis_refcoco_updated+nv_metropolis_vqav2_updated+nv_metropolis_aokvqa_legal+gqa_commercial_updated+aokvqa_commercial+synthdog_en+wit_subset+clevr_merge_explanation+clevr_math+oasst_processed_laion_openasst+vila_nvidia_qa_augment_3x+vila_nvidia_from_oasst+processed_video_v1+df_human_v1+df_human_v1_qa+vatex_cc+youcook2_cc+vcg_human_cc+vidln_cc+reason_clevrerqa+perception_test+nextqa_mc_cc+reason_clevrermc+kinetics400_cc+eagle_commercial \
  --vila-datasets cs-oci-ord.yaml  \
  --image-aspect-ratio dynamic --max-tiles 6 \
  --mm-projector mlp2x_tome2d_w48_h48_sx6_sy6_r2108 \
  --stage3-learning-rate 2e-5

# Quick Training on IAD
./run_job.py   --depchain 1 --nodes 8  --image /lustre/fsw/portfolios/llmservice/users/gheinrich/cache/vila-7591ca.sqsh \
  --output /lustre/fsw/portfolios/llmservice/users/gheinrich/results/25-11-24-vila-radiodino  \
  ./scripts/experimental/radio/train-all.sh  \
  --conda-path /lustre/fsw/portfolios/llmservice/users/gheinrich/anaconda3/bin/conda \
  --vision-tower  radio:768:gheinrich/RADIO-DINOv2-g \
  --stage2-dataset sharegpt4v_pretrain+coyo_25m_wds_spatial_ocr_bbox_interleaved_qas+docmatix_750k+mmc4core_10_subset \
  --stage3-dataset sharegpt4v_gpt4_100k+llava_instruct+sharegpt4v_sft+dvqa_train_200k+chartqa_train_18k+ai2d_train_12k+docvqa_train_10k+geoqa+synthdog_en \
  --vila-datasets draco-oci-iad.yaml  \
  --image-aspect-ratio dynamic --max-tiles 6 \
  --mm-projector mlp2x_tome2d_w48_h48_sx6_sy6_r2108 \
  --stage3-learning-rate 2e-5
'

STAGE1_BATCH_SIZE=2048
STAGE1_GRADIENT_ACCUMULATION_STEPS=1
STAGE1_DATASET="llava_1_5_mm_align"
STAGE1_LEARNING_RATE=1e-3


STAGE2_BATCH_SIZE=1024
STAGE2_GRADIENT_ACCUMULATION_STEPS=2
STAGE2_DATASET="sharegpt4v_pretrain"
STAGE2_LEARNING_RATE=5e-5

STAGE3_BATCH_SIZE=2048
STAGE3_GRADIENT_ACCUMULATION_STEPS=16
STAGE3_DATASET="sharegpt4v_sft+vflan"
STAGE3_LEARNING_RATE=2e-5

MODEL_MAX_LENGTH=4096

MM_VISION_SELECT_FEATURE="cls_patch"
MM_VISION_SELECT_LAYER=-1
MM_PROJECTOR=mlp_downsample
VISION_TOWER="radio:768:nvidia/RADIO-L"
IMAGE_ASPECT_RATIO="resize"
MODEL="nvidia/Mistral-NeMo-Minitron-8B-Base"
CHAT_TEMPLATE="mistral"

OUTPUT_DIR=/tmp
CHAT_TEMPLATE="mistral"
MAX_TILES=12

REPORT_TO="wandb"
VILA_DATASETS=draco-oci-iad.yaml

STAGE3_TUNE_VISION_TOWER=True

CONDA_PATH="/lustre/fsw/portfolios/nvr/users/gheinrich/anaconda3/bin/conda"

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output)
      OUTPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    --conda-path)
      CONDA_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -r|--report-to)
        REPORT_TO="$2"
        shift # past argument
        shift # past value
        ;;
    --stage1-path)
        STAGE1_OUTPUT_DIR="$2"
        shift # past argument
        shift # past value
        ;;
    --stage2-path)
        STAGE2_OUTPUT_DIR="$2"
        shift # past argument
        shift # past value
        ;;
    --stage1-batch-size)
        STAGE1_BATCH_SIZE="$2"
        shift # past argument
        shift # past value
        ;;
    --stage2-batch-size)
        STAGE2_BATCH_SIZE="$2"
        shift # past argument
        shift # past value
        ;;
    --stage3-batch-size)
        STAGE3_BATCH_SIZE="$2"
        shift # past argument
        shift # past value
        ;;
    --stage1-dataset)
        STAGE1_DATASET="$2"
        shift # past argument
        shift # past value
        ;;
    --stage2-dataset)
        STAGE2_DATASET="$2"
        shift # past argument
        shift # past value
        ;;
    --stage3-dataset)
        STAGE3_DATASET="$2"
        shift # past argument
        shift # past value
        ;;
    --stage1-learning-rate)
        STAGE1_LEARNING_RATE="$2"
        shift # past argument
        shift # past value
        ;;
    --stage2-learning-rate)
        STAGE2_LEARNING_RATE="$2"
        shift # past argument
        shift # past value
        ;;
    --stage3-learning-rate)
        STAGE3_LEARNING_RATE="$2"
        shift # past argument
        shift # past value
        ;;
    --stage1-gradient-accumulation-steps)
        STAGE1_GRADIENT_ACCUMULATION_STEPS="$2"
        shift # past argument
        shift # past value
        ;;
    --stage2-gradient-accumulation-steps)
        STAGE2_GRADIENT_ACCUMULATION_STEPS="$2"
        shift # past argument
        shift # past value
        ;;
    --stage3-gradient-accumulation-steps)
        STAGE3_GRADIENT_ACCUMULATION_STEPS="$2"
        shift # past argument
        shift # past value
        ;;
    --stage3-tune-vision-tower)
        STAGE3_TUNE_VISION_TOWER="$2"
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
    --max-tiles)
        MAX_TILES="$2"
        shift # past argument
        shift # past value
        ;;
    --vila-datasets)
        VILA_DATASETS="$2"
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

source ~/.bashrc

# If Stage1 output directories is not provided, default to OUTPUT_DIR/stage1.
STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-$OUTPUT_DIR/stage1}
# If Stage2 output directories is not provided, default to OUTPUT_DIR/stage2.
STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-$OUTPUT_DIR/stage2}
STAGE3_OUTPUT_DIR=$OUTPUT_DIR/stage3

# enable user's conda
eval "$(${CONDA_PATH} shell.bash hook)"

conda run --no-capture-output  -n vila pip install hydra-core
conda run --no-capture-output  -n vila pip install loguru

export DEEPSPEED_LOG_LEVEL=DEBUG

export VILA_DATASETS=${VILA_DATASETS}

if [ ! -f "$STAGE1_OUTPUT_DIR/model/config.json" ]; then
    echo "Running Stage 1"

    DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=${STAGE1_BATCH_SIZE}
    DEFAULT_GRADIENT_ACCUMULATION_STEPS=${STAGE1_GRADIENT_ACCUMULATION_STEPS}
    DEFAULT_RUN_NAME=$(basename $OUTPUT_DIR)-stage1

    source scripts/setups/train.sh

    conda run --no-capture-output  -n vila torchrun \
        --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        llava/train/train_mem.py \
            --deepspeed scripts/zero3.json \
            --model_name_or_path ${MODEL} \
            --chat_template ${CHAT_TEMPLATE} \
            --data_mixture ${STAGE1_DATASET} \
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
            --max_tiles ${MAX_TILES} \
            --bf16 True \
            --output_dir $STAGE1_OUTPUT_DIR/model \
            --num_train_epochs 1 \
            --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps $STAGE1_GRADIENT_ACCUMULATION_STEPS \
            --save_strategy steps \
            --save_steps 100 \
            --save_total_limit 1 \
            --learning_rate ${STAGE1_LEARNING_RATE} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --model_max_length ${MODEL_MAX_LENGTH} \
            --gradient_checkpointing True \
            --dataloader_num_workers 16 \
            --report_to ${REPORT_TO}
elif [ ! -f "$STAGE2_OUTPUT_DIR/model/config.json" ]; then
    echo "Running Stage 2"
    DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=${STAGE2_BATCH_SIZE}
    DEFAULT_GRADIENT_ACCUMULATION_STEPS=${STAGE2_GRADIENT_ACCUMULATION_STEPS}
    DEFAULT_RUN_NAME=$(basename $OUTPUT_DIR)-stage2

    source scripts/setups/train.sh

    conda run --no-capture-output  -n vila torchrun \
        --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        llava/train/train_mem.py \
            --deepspeed scripts/zero3.json \
            --model_name_or_path ${STAGE1_OUTPUT_DIR}/model \
            --data_mixture ${STAGE2_DATASET} \
            --vision_tower ${VISION_TOWER} \
            --mm_vision_select_feature ${MM_VISION_SELECT_FEATURE} \
            --mm_projector ${MM_PROJECTOR} \
            --tune_vision_tower False \
            --tune_mm_projector True \
            --tune_language_model True \
            --mm_vision_select_layer ${MM_VISION_SELECT_LAYER} \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio ${IMAGE_ASPECT_RATIO} \
            --max_tiles ${MAX_TILES} \
            --bf16 True \
            --output_dir $STAGE2_OUTPUT_DIR/model \
            --num_train_epochs 1 \
            --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps $STAGE2_GRADIENT_ACCUMULATION_STEPS \
            --save_strategy steps \
            --save_steps 100 \
            --save_total_limit 1 \
            --learning_rate ${STAGE2_LEARNING_RATE} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --model_max_length ${MODEL_MAX_LENGTH} \
            --gradient_checkpointing True \
            --dataloader_num_workers 16 \
            --report_to ${REPORT_TO}
elif [ ! -f "$STAGE3_OUTPUT_DIR/model/config.json" ]; then
    echo "Running Stage 3"
    DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=${STAGE3_BATCH_SIZE}
    DEFAULT_GRADIENT_ACCUMULATION_STEPS=${STAGE3_GRADIENT_ACCUMULATION_STEPS}
    DEFAULT_RUN_NAME=$(basename $OUTPUT_DIR)-stage3

    source scripts/setups/train.sh

    conda run --no-capture-output  -n vila torchrun \
        --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        llava/train/train_mem.py \
            --deepspeed scripts/zero3.json \
            --model_name_or_path ${STAGE2_OUTPUT_DIR}/model \
            --data_mixture ${STAGE3_DATASET} \
            --mm_projector ${MM_PROJECTOR} \
            --tune_vision_tower ${STAGE3_TUNE_VISION_TOWER} \
            --tune_mm_projector True \
            --tune_language_model True \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio ${IMAGE_ASPECT_RATIO} \
            --max_tiles ${MAX_TILES} \
            --bf16 True \
            --output_dir $STAGE3_OUTPUT_DIR/model \
            --num_train_epochs 1 \
            --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps $STAGE3_GRADIENT_ACCUMULATION_STEPS \
            --save_strategy steps \
            --save_steps 50 \
            --save_total_limit 1 \
            --learning_rate ${STAGE3_LEARNING_RATE} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type cosine \
            --logging_steps 1 \
            --model_max_length ${MODEL_MAX_LENGTH} \
            --gradient_checkpointing True \
            --dataloader_num_workers 16 \
            --vflan_no_system_prompt True \
            --report_to ${REPORT_TO}
else
    echo "All done!"
fi
