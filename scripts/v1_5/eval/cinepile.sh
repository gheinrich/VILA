#!/bin/bash
MODEL_PATH=$1
CKPT=$2
CONV_MODE=vicuna_v1
if [ "$#" -ge 3 ]; then
    CONV_MODE="$3"
fi

output_dir="./eval_output/$CKPT/CinePile"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

echo "Evaluating $CKPT with conv_mode $CONV_MODE..."
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 llava/eval/model_vqa_cinepile.py \
    --model-path $MODEL_PATH \
    --temperature 0 \
    --conv-mode $CONV_MODE \
    --output_dir ${output_dir} \
    --output_name ${CHUNKS}_${IDX} \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX &
done

wait

output_file=${output_dir}/merge.json

# Clear out the output file if it exists.
if [ -f "$output_file" ]; then
    > "$output_file"
fi


# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done

# compute accuracy
python scripts/v1_5/eval/compute_accuracy.py --output_dir ${output_dir}