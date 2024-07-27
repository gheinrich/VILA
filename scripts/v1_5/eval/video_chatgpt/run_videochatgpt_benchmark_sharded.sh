#!/bin/bash
set -e

model_path=$1 # first argument is the model path
ckpt_name=$2 # second argument is the evaluation output directory name
conv_mode=vicuna_v1
result_dir=./eval_output/${ckpt_name}/videochatgpt
if [ "$#" -ge 3 ]; then
    conv_mode="$3"
fi


# general
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=$(( ${#GPULIST[@]} / 2 )) # Calculate chunks for 2 GPUs per chunk

BENCHMARK=/home/shijial/workspace/LITA-1.5/data/evaluation/video_chatgpt/benchmarking
VIDEO_DIR="${BENCHMARK}/Test_Videos"

function model_videochatgpt_benchmark_sharded {
    gt_file=${1}
    output_dir=${2}
    if [[ -f "$output_dir/merge.jsonl" ]];
    then
        return
    fi
    echo "running ${gt_file}, output to ${output_dir}"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        GPU_IDX1=$((IDX * 2))  # First GPU index
        GPU_IDX2=$((GPU_IDX1 + 1))  # Second GPU index
        CUDA_VISIBLE_DEVICES=${GPULIST[$GPU_IDX1]},${GPULIST[$GPU_IDX2]} python llava/eval/model_videochatgpt_benchmark.py \
            --model-path ${model_path} \
            --image-folder ${VIDEO_DIR} \
            --gt_file ${gt_file} \
            --output_dir ${output_dir} \
            --output_name ${CHUNKS}_${IDX} \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --conv-mode $conv_mode &
    done

    wait

    output_file=${output_dir}/merge.jsonl

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${output_dir}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
}

# general
gt_file="${BENCHMARK}/Benchmarking_QA/generic_qa.json"
output_dir="${result_dir}/generic_qa"
model_videochatgpt_benchmark_sharded ${gt_file} ${output_dir}

# temporal
gt_file="${BENCHMARK}/Benchmarking_QA/temporal_qa.json"
output_dir="${result_dir}/temporal_qa"
model_videochatgpt_benchmark_sharded ${gt_file} ${output_dir}

# consistency
gt_file="${BENCHMARK}/Benchmarking_QA/consistency_qa.json"
output_dir="${result_dir}/consistency_qa"
model_videochatgpt_benchmark_sharded ${gt_file} ${output_dir}
