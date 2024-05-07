#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/${USER}/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export DECORD_DUPLICATE_WARNING_THRESHOLD=1.0
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
    --model_name_or_path ./checkpoints/vicuna-13b-siglipso400m-pretrain-ccsvideo-coyo_25m_mmc4core_sharegpt4v_internvid_10M-linear-e11113 \
    --version v1 \
    --data_mixture vflan+sharegpt4v_sft+video_chatgpt+youcook2+vatex+activitynet_qa+ivqa+nextqa+msrvttqa+jukinmedia+shot2story_shotonly \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector mlp2x_gelu \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/vicuna-13b-siglipso400m-ccsvideo-coyo_25m_mmc4core_sharegpt4v_internvid_10M-finetune-baseline_nv_video_flan_jukin_shot2story_shot_only-e11113 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb
