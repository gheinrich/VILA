source ~/.bashrc
conda activate vila-internal
# multi-node
export WANDB_RESUME="allow"
export WANDB_PROJECT="VILA-SR"
export WANDB_RUN_ID="test_new_codebase"
export WANDB_API_KEY="7f6b27896804432690638644d3687ffa5000d9c6"

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
echo "MASTER_ADDR="$MASTER_ADDR

worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
echo "JobID: $SLURM_JOB_ID Full worker list: $worker_list"

n_node=$SLURM_JOB_NUM_NODES
acc_step=${ACC_STEP:-1}
bs=$((32 / n_node / acc_step))

echo "number of nodes:" $n_node
echo "accmulation steps:" $acc_step
echo "per device batch size :" $bs
echo "node rank:" $CURRENT_RANK

LOAD_CKPT=/home/yunhaof/workspace/ckpts/llava-v1.5-7b

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25000 \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LOAD_CKPT \
    --version v1 \
    --data_mixture vflan+sharegpt4v_sft \
    --vision_tower google/siglip-large-patch16-384 \
    --vision_resolution 576 \
    --mm_projector_typemlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --tune_mm_projector True \
    --tune_vision_tower True \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/yunhaof/workspace/projects/VILA-Internal/checkpoints/super_resolution/sam/stage2.5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
