source ~/.bashrc
conda activate vila
which python

WANDB_RESUME=allow
# cd ~/workspace/multi-modality-research/VILA/

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}

worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
echo "JobID: $SLURM_JOB_ID Full worker list: $worker_list"
echo "MASTER_ADDR="$MASTER_ADDR
n_node=${SLURM_JOB_NUM_NODES:-1}

bs=$((128 / n_node))
# bs=4
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $SLURM_PROCID

DATASET=${DATASET:-"coyo_25m_refilter+mmc4core"}

echo $DATASET

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    llava/train/train_mem.py \
    --model_name_or_path /home/jasonlu/models/vicuna-1.5/vicuna-7b-v1.5 \
    --version v1 \
    --datasets_mixture_name $DATASET \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type linear \
    --pretrain_mm_mlp_adapter checkpoints/vicuna-7b-clip336-pretrain-ccs-linear-e1/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --bf16 True \
    --output_dir ./checkpoints/vicuna-7b-clip336-finetune-$DATASET-linear-e8 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb
