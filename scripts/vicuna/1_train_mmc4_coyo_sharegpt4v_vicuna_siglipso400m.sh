source ~/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python

cd /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

n_node=$SLURM_JOB_NUM_NODES
bs=$((256 / n_node))
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "total batch size:" $((bs * 8 * n_node))
echo "node rank:" $SLURM_PROCID


torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    llava/train/train_mem.py \
    --model_name_or_path /lustre/fsw/portfolios/nvr/users/jasonlu/models/vicuna-1.5/vicuna-7b-v1.5 \
    --version v1 \
    --datasets_mixture_name coyo_25m_mmc4core_sharegpt4v \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type linear \
    --pretrain_mm_mlp_adapter /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/ckpts/vicuna-7b-siglipso400m-pretrain-ccs-linear-e1/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --bf16 True \
    --output_dir /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/ckpts/vicuna-7b-siglipso400m-pretrain-ccs-linear-e11 \
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
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 32 \
    --lazy_preprocess True \
    --report_to wandb