#!/bin/bash
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

n_node=$SLURM_JOB_NUM_NODES
# bs=$((128 / n_node / 8 / 2))
bs=1
echo "number of nodes:" $n_node
echo "per device batch size:" $bs
echo "node rank:" $SLURM_PROCID

# Parse command-line arguments for vt and llm
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --vt) vt_mode="$2"; shift ;;
        --llm) llm_mode="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check for valid modes
valid_modes=("freeze" "ft" "lora")
if [[ ! " ${valid_modes[@]} " =~ " ${vt_mode} " ]]; then
    echo "Invalid vt mode: $vt_mode. Valid options are: ${valid_modes[*]}"
    exit 1
fi
if [[ ! " ${valid_modes[@]} " =~ " ${llm_mode} " ]]; then
    echo "Invalid llm mode: $llm_mode. Valid options are: ${valid_modes[*]}"
    exit 1
fi

# Set configurations based on vt_mode
tune_vision_tower="False"
lora_vt="False"
if [[ "$vt_mode" == "lora" ]]; then
    lora_vt="True"
    tune_vision_tower="True"
elif [[ "$vt_mode" == "ft" ]]; then
    tune_vision_tower="True"
elif [[ "$vt_mode" == "freeze" ]]; then
    tune_vision_tower="False"
fi

# Set configurations based on llm_mode
tune_language_model="False"
lora_llm="False"
if [[ "$llm_mode" == "lora" ]]; then
    lora_llm="True"
    tune_language_model="True"
elif [[ "$llm_mode" == "ft" ]]; then
    tune_language_model="True"
elif [[ "$llm_mode" == "freeze" ]]; then
    tune_language_model="False"
fi

# Add --lora_enable if any module uses lora
lora_enable=""
if [[ "$lora_vt" == "True" ]] || [[ "$lora_llm" == "True" ]]; then
    lora_enable="--lora_enable"
fi


# --lora_x apply lora on x, if not --lora_x, --tune_x enable ft for x
torchrun --nnodes=$n_node --nproc_per_node=8 --master_port=25001 \
    --master_addr $MASTER_ADDR --node_rank=$SLURM_PROCID \
    llava/train/train_mem.py \
    $lora_enable \
    --lora_llm $lora_llm \
    --lora_vt $lora_vt \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path  Efficient-Large-Model/Llama-3-VILA1.5-8B\
    --version llama_3 \
    --data_mixture scienceqa_mix \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower $tune_vision_tower \
    --tune_mm_projector True \
    --tune_language_model $tune_language_model \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_dir\
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3 \
    --save_total_limit 1 \
    --max_steps 5 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
