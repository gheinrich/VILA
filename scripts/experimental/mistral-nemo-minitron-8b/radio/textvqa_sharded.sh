#!/bin/bash


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=$(( ${#GPULIST[@]} / 2 )) # Calculate chunks for 2 GPUs per chunk

MODEL_PATH=/specify/on/command/line
DATA_DIR=/home/yunhaof/workspace/datasets/evaluation
CONV_MODE=vicuna_v1

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output)
      OUTPUT_DIR="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--model-path)
        MODEL_PATH="$2"
        shift # past argument
        shift # past value
        ;;
    --conv-mode)
        CONV_MODE="$2"
        shift # past argument
        shift # past value
        ;;
    *)
      shift # past argument
      ;;
  esac
done

mkdir -p ${OUTPUT_DIR}/textvqa/answers/

# Overwrite Transformers
cp -r llava/train/transformers_replace/* /opt/conda/envs/vila/lib/python3.10/site-packages/transformers/

for IDX in $(seq 0 $((CHUNKS-1))); do
    GPU_IDX1=$((IDX * 2))  # First GPU index
    GPU_IDX2=$((GPU_IDX1 + 1))  # Second GPU index

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX1]},${GPULIST[$GPU_IDX2]} conda run --no-capture-output -n vila python llava/eval/model_vqa_loader.py \
      --model-path $MODEL_PATH \
      --question-file ${DATA_DIR}/textvqa/llava_textvqa_val_v051_ocr.jsonl \
      --image-folder ${DATA_DIR}/textvqa/train_images \
      --answers-file ${OUTPUT_DIR}/textvqa/answers/${CHUNKS}_${IDX}.jsonl \
      --generation-config '{"max_new_tokens": 128}' \
      --num-chunks $CHUNKS \
      --chunk-idx $IDX \
      --conv-mode ${CONV_MODE} &
done

wait

output_file=${OUTPUT_DIR}/textvqa/answers/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    # add newline if needed
    shard_file="${OUTPUT_DIR}/textvqa/answers/${CHUNKS}_${IDX}.jsonl"
    sed -i -e '$a\' ${shard_file}
    cat ${shard_file} >> "$output_file"
done

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n vila python llava/eval/eval_textvqa.py \
    --annotation-file ${DATA_DIR}/textvqa/TextVQA_0.5.1_val.json \
    --result-file $output_file
