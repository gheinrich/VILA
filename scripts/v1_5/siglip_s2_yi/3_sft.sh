#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/baifengs/projects/multi-modality-research/VILA/

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

n_node=$SLURM_JOB_NUM_NODES
bs=$((128 / n_node))
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $SLURM_PROCID

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /home/jasonlu/workspace/multi-modality-research/VILA/checkpoints/vila-yi-34b-siglip-alignment_fix800_draco \
    --version hermes-2 \
    --data_mixture sharegpt4v_gpt4_100k+llava_instruct+sharegpt4v_sft+dvqa_train_200k+chartqa_train_18k+ai2d_train_12k+docvqa_train_10k+geoqa+synthdog_en \
    --vision_tower google/siglip-so400m-patch14-384 \
    --s2 True \
    --s2_scales "384,768,1152" \
    --s2_max_split_size 384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp2x_gelu \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --drop_path_rate 0.4 \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./output/vila-yi-34b-siglip-s2-alignment_fix801_draco \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
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
