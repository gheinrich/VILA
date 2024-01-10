source ~/.bashrc
conda activate vila
which python

cd ~/workspace/multi-modality-research/VILA/
rm -r ~/workspace/multi-modality-research/VILA/checkpoints/mixtral-7b-clip336-finetune-mmc4sub+coyo-linear-e4_test

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

n_node=$SLURM_JOB_NUM_NODES
bs=$((1 / n_node))
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $SLURM_PROCID


torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    llava/train/train_mem.py \
    --model_name_or_path /home/jasonlu/models/Mixtral-8x7B-v0.1 \
    --version v1 \
    --datasets_mixture_name coyo_25m_mmc4core_test \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type linear \
    --pretrain_mm_mlp_adapter checkpoints/mixtral-8x7b-clip336-pretrain-ccs-linear-e3/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --bf16 True \
    --output_dir ./checkpoints/mixtral-7b-clip336-finetune-mmc4sub+coyo-linear-e4_test \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --gradient_accumulation_steps 1 \
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
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'MixtralDecoderLayer' \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb
