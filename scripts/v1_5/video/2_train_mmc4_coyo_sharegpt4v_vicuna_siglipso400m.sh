#!/bin/bash
source /lustre/fsw/portfolios/nvr/users/${USER}/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA-Internal

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export DECORD_DUPLICATE_WARNING_THRESHOLD=1.0
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

n_node=$SLURM_JOB_NUM_NODES
bs=$((256 / n_node))
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $SLURM_PROCID

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /home/jasonlu/models/vicuna-1.5/vicuna-7b-v1.5 \
    --version v1 \
    --data_mixture coyo_25m+mmc4core+sharegpt4v_pretrained+valley \
    --vision_tower google/siglip-so400m-patch14-384 \
    --pretrain_mm_mlp_adapter /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/ckpts/vicuna-7b-siglipso400m-pretrain-ccs-linear-e1/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/ckpts/vicuna-7b-siglipso400m-pretrain-ccs-coyo_25m_mmc4core_sharegpt4v_valley-linear-e111 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 270 \
    --save_total_limit 1 \
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
    --report_to wandb
