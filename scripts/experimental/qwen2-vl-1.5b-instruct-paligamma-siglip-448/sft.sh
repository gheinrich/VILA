#!/bin/bash

DEFAULT_RUN_NAME="vila-qwen2-1.5b-instruct-paligemma-siglip-448-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=2048
DEFAULT_GRADIENT_ACCUMULATION_STEPS=4

source scripts/setups/train.sh

STAGE2_PATH=$1

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3_gradient_clipping.json \
        --model_name_or_path $STAGE2_PATH \
        --data_mixture sharegpt4v_gpt4_100k+llava_instruct+sharegpt4v_sft+dvqa_train_200k+chartqa_train_18k+ai2d_train_12k+docvqa_train_10k+geoqa+synthdog_en+scienceqa+wit_subset+math+sherlock+idefics2_sft+llave_onevision_images_sft+cambrian_1375k+shot2story_shotonly+video_chatgpt+youcook2+vatex+sharegpt_video+stem_qa+nv_mm_sft+jukinmedia+sharegpt4video+k710+ssv2+reason_clevrerqa+reason_clevrermc+vcg_human+video_chat1+av_llava_4785+vflan+refcoco_train+shikra+lrv_instruction+textocr_qa+mmc_instruction+m4-instruct-video+nextqa_mc+unimm_chat+svit+mmbench_val+cvbench+m4-instruct-image-nuscenes+webvid_qa+caption_videochat+doc_reason+metamathqa+mminstruct+unichart+mtwi+kvqa \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_2x2_fix \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
        --max_grad_norm 5.0 \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to wandb
