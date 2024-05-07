#!/bin/bash

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal

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
    --model_name_or_path /home/jasonlu/workspace/VILA-Internal/checkpoints/vila-intern-yi-34b-r401 \
    --version hermes-2 \
    --data_mixture sharegpt4v_gpt4_100k+llava_instruct+sharegpt4v_sft+dvqa_train_200k+chartqa_train_18k+ai2d_train_12k+docvqa_train_10k+geoqa+synthdog_en+vflan+shot2story_shotonly+video_chatgpt+youcook2+vatex+scienceqa+wit_subset+math+sherlock \
    --vision_tower /home/jasonlu/models/InternViT-6B-448px-V1-2 \
    --mm_projector mlp_downsample \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -1 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/vila-intern-yi-34b-r402 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb