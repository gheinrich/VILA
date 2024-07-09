MODEL=$1
TYPE=$2
pred_path=$3
Video_5_Benchmark="eval_vila_benchmark"
output_dir="${Video_5_Benchmark}/${MODEL}/gpt4/${TYPE}"
output_json="${Video_5_Benchmark}/${MODEL}/results/${TYPE}_qa.json"

mkdir -p "${Video_5_Benchmark}/${MODEL}/results/"

python llava/eval/video/eval_benchmark_${TYPE}.py \
    --pred_path  ${pred_path} \
    --output_dir  ${output_dir} \
    --output_json  ${output_json} \
    --num_tasks 8
