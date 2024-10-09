#!/bin/bash

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
#export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
#export CUDA_LAUNCH_BLOCKING=1

###########################################################################
# cluster related information
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"
###########################################################################
export VISION_TOWER=${VISION_TOWER:-"google/siglip-large-patch16-384"}
# google/siglip-large-patch16-384
# openai/clip-vit-large-patch14-336

# GLOBAL bs: 128 * 8
export ALIGN_DATASET=${ALIGN_DATASET:-llava_1_5_mm_align}
export PT_DATASET=${PT_DATASET:-sharegpt4v_pretrain}
export SFT_DATASET=${SFT_DATASET:-sharegpt4v_sft+vflan}
export SEED=${SEED:-42}

sort_and_join() {
    local original_string=$1
    local delimiter=$2
    # Save the current IFS
    local oldIFS=$IFS
    # Split the string into an array based on the delimiter
    IFS="$delimiter" read -r -a array <<< "$original_string"
    # Sort the array
    sorted_array=($(for i in "${array[@]}"; do echo "$i"; done | sort))
    # Concatenate the sorted array elements back into a string
    IFS="$delimiter"; sorted_string="${sorted_array[*]}"
    # Restore the original IFS
    IFS=$oldIFS
    # Return the sorted, concatenated string
    echo "$sorted_string"
}

delimiter="+"
ALIGN_DATASET=$(sort_and_join "$ALIGN_DATASET" "$delimiter")
PT_DATASET=$(sort_and_join "$PT_DATASET" "$delimiter")
SFT_DATASET=$(sort_and_join "$SFT_DATASET" "$delimiter")

echo "ALIGN: $ALIGN_DATASET | PRETRAIN: $PT_DATASET | SFT: $SFT_DATASET"

global_bs=${BATCH_SIZE:-128}
acc_step=${ACC_STEP:-1}
bs=$((global_bs / n_node / acc_step))

if [ "$n_node" = "1" ]; then
    #FIXME: set an extra to surprass the setting.
    echo "Detected on single machine. Automatically set batch size to 32 for debugging purpose."
    bs=32
fi

export BASE_MODEL_PATH=${BASE_MODEL_PATH:-"NousResearch/Llama-2-7b-hf"}
MNAME=$(echo $BASE_MODEL_PATH | rev | cut -d "/" -f 1 | rev)
VTOWER=$(echo $VISION_TOWER | rev | cut -d "/" -f 1 | rev)
# OUTPUT_STEP1=${1:-"$MNAME-$VISION_TOWER-align-$ALIGN_DATASET"}
# OUTPUT_STEP2=${2:-"./checkpoints/$MNAME-$VTOWER-align-$ALIGN_DATASET-pretrain-$PT_DATASET"}
# OUTPUT_STEP3=${3:-"./checkpoints/$MNAME-$VTOWER-align-$ALIGN_DATASET-pretrain-$PT_DATASET-SFT-$SFT_DATASET"}

STAGE2_PATH=$1
OUTPUT_DIR=${OUTPUT_DIR:-"runs/dev"}

echo "[vision] $VISION_TOWER \n[loading] from $OUTPUT_STEP2, \n[saving] to $OUTPUT_STEP3"

echo "number of nodes:" $n_node
echo "per device batch size: $bs | global batch size $global_bs"
echo "node rank:" $CURRENT_RANK
echo "ALIGN: $ALIGN_DATASET | PRETRAIN: $PT_DATASET | SFT: $SFT_DATASET"

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
   --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $STAGE2_PATH \
    --version v1 \
    --data_mixture $SFT_DATASET \
    --vision_tower $VISION_TOWER \
    --mm_projector mlp2x_gelu \
    --tune_language_model True \
    --tune_mm_projector True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --group_by_modality_length True \
    --bf16 True \
    --seed $SEED \
    --output_dir $OUTPUT_DIR/model \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $acc_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 210 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5120 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb \
    --quantize_model "te_qlinear" \
    --fabit "E4M3" \
    --fwbit "E4M3" \
    --bobit "E5M2" \
    --row_blocksize -1 \
    --col_blocksize -1 \
    --pad_to_multiple_of 4

# SFT_DATASET=shot2story_shotonly bash scripts/v1_5/caption/3_sft_captioner.sh /home/ligengz/workspace/video_checkpoint/video-13b
