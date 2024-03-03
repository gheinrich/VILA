
LOAD_CKPT=/home/yunhaof/workspace/ckpts/llava-v1.5-7b

torchrun --nproc_per_node=1 --master_port=25000 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path $LOAD_CKPT \
    --version v1 \
    --data_mixture llava_1_5_mm_align \
    --vision_tower google/siglip-large-patch16-384 \
    --vision_resolution 576 \
    --vision_projector mlp2x_gelu \
    --tune_language_model True \
    --tune_vision_projector True \
    --vision_select_layer -2 \
    --vision_select_feature patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True