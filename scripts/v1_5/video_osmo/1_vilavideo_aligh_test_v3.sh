#!/bin/bash
source ~/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python

cd ~/VILA
echo "MASTER_ADDR="$MASTER_ADDR

n_node=$WORLD_SIZE
seq_parallel_size=8
# bs=$((128 * seq_parallel_size / n_node))
bs=2
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $NODE_RANK

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$NODE_RANK \
    llava/train/train_hybrid.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /mnt/amlfs-01/home/fuzhaox/checkpoints/Meta-Llama-3-8B-Instruct \
    --version llama_3 \
    --data_mixture osmo_sharegpt4v_sft+osmo_sharegpt_video_qa \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector mlp_downsample \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature cls_patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/vilavideo8b_align_test_v3_ga8_sp8_2nodes_loss_debug_v3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --num_video_frames 8 \
    --fps 2.0 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --seq_parallel_size $seq_parallel_size