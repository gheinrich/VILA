source ~/.bashrc
conda activate llavadepfix
which python

if [ ! -d "/tmp/llava-1.5" ]; then
    echo "Preparing dataset..."
    mkdir -p /tmp/llava-1.5
    # /tmp/llava-1.5/coco/train2017
    echo "unzip coco"
    mkdir -p /tmp/llava-1.5/coco
    unzip -q /home/jil/datasets/llava-1.5/coco/train2017.zip -d /tmp/llava-1.5/coco/ &

    # /tmp/llava-1.5/gqa/images
    echo "unzip gqa"
    mkdir -p /tmp/llava-1.5/gqa
    unzip -q /home/jil/datasets/llava-1.5/gqa/images.zip -d /tmp/llava-1.5/gqa/ &

    # /tmp/llava-1.5/ocr_vqa/images
    echo "unzip ocr_vqa"
    mkdir -p /tmp/llava-1.5/ocr_vqa
    tar -xf /home/jil/datasets/llava-1.5/ocr_vqa/images.tar --directory /tmp/llava-1.5/ocr_vqa/ &

    # /tmp/llava-1.5/textvqa/train_images/
    echo "unzip textvqa"
    mkdir -p /tmp/llava-1.5/textvqa
    unzip -q /home/jil/datasets/llava-1.5/textvqa/train_val_images.zip -d /tmp/llava-1.5/textvqa/ &

    # vg
    echo "unzip vg"
    mkdir -p /tmp/llava-1.5/vg
    unzip -q /home/jil/datasets/llava-1.5/vg/images.zip -d /tmp/llava-1.5/vg/ &
    unzip -q /home/jil/datasets/llava-1.5/vg/images2.zip -d /tmp/llava-1.5/vg/ &

    wait  # finish all unpacking
    # move the dir
    mv /tmp/llava-1.5/ocr_vqa/tmp/llava-1.5/ocr_vqa/images/ /tmp/llava-1.5/ocr_vqa/images

    echo "done"

    echo "old data as well"
    echo "Preparing dataset..."
    tar -xf ~/datasets/coco.tar --directory /tmp/ &

    tar -xf ~/datasets/lrv-instruct/vg-images.tar.gz --directory /tmp/ &

    wait
    # mv /tmp/image/* /tmp/coco/train2017/
    find /tmp/image/ -name '*.*' -exec mv -t /tmp/coco/train2017/ {} +

    mkdir /tmp/coco/train2017/coco/
    ln -s /tmp/coco/train2017/ /tmp/coco/train2017/coco/

    echo "done"
else
    echo "Data already exists..."
fi

cd ~/workspace/LLaVA/

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

n_node=$SLURM_JOB_NUM_NODES
bs=$((64 / n_node))
n_gpus=$((n_node * 8))
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $SLURM_PROCID

torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    llava/train/train_mem.py \
    --model_name_or_path /home/jil/workspace/LLaVA/checkpoints/llama2-7b-finetune-mmc4sub-linear-e1-nose-run2-visexp-visattn \
    --version v1 \
    --data_path /home/jil/datasets/vlm-flan-clean-text1m-nosqa \
    --image_folder /tmp/coco/train2014 \
    --dataset_type vflan \
    --aug_llava True \
    --aug_llava_path /home/jil/datasets/llava-1.5/llava_v1_5_mix665k.json \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector linear \
    --min_max_range_path /home/jil/models/llama-2-hf/llama-2-7b/emb_min_max.pt \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --bf16 True \
    --output_dir ./checkpoints/llama2-7b-mmc4sub-finetune-llava15+vflan-nosqa-linear-e1-nose-run2-visexp-visattn \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 210 \
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
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
