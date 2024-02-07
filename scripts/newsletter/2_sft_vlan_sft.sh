source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA/

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}

worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
echo "JobID: $SLURM_JOB_ID Full worker list: $worker_list"
echo "MASTER_ADDR="$MASTER_ADDR
n_node=${SLURM_JOB_NUM_NODES:-1}
acc_step=${ACC_STEP:-1}
bs=$((512 / n_node / acc_step))
# bs=4
echo "number of nodes:" $n_node
echo "accmulation steps:" $acc_step
echo "per device batch size :" $bs
echo "node rank:" $SLURM_PROCID


PT_DATASET=${PT_DATASET:-"mmc4core"}
DATASET=${DATASET:-"vflan_sharegpt4v_sft"}

echo "PT dataset: " $PT_DATASET
echo "dataset: " $DATASET

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    llava/train/train_mem.py \
    --model_name_or_path ./checkpoints/vicuna-7b-clip336-finetune-$PT_DATASET-linear-e8 \
    --version v1 \
    --datasets_mixture_name $DATASET \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type linear \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --bf16 True \
    --output_dir ./checkpoints/ablation-vicuna-7b-clip336-PT:$PT_DATASET-SFT:$DATASET \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --gradient_accumulation_steps $acc_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 150 \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5120 \
    --gradient_checkpointing True \
    --dataloader_num_workers 24 \
    --lazy_preprocess True \
    --report_to wandb \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
