#/bin/bash
set -e

# export VILA_SLURM_ACCOUNT=llmservice_nlp_fm
export DATA_MIXTURE=${DATA_MIXTURE:-"sharegpt4v_sft"}

nodes=${1:-"8"}
# fp16, fp8, fp8_memoryefficient
# qwen_fp8, qwen_fp16

TASK=${TASK:-"sft"}
# data_name=${DATA_MIXTURE/+/_}
mname=j63-$TASK-nodes_$nodes
CONV_MODE="auto"

if [ "$nodes" = "1" ] || [ "$nodes" = "2" ]; then
    VILA_SLURM_PARTITION=$VILA_SLURM_PARTITION,interactive
fi

# export WANDB_PROJECT="vila-fp8-debug"
export TRITON_HOME=/home/ligengz/.cache/trion

if [ ! -f runs/train/$mname/model/config.json ]; then
    vila-run -m train -J $mname -N $nodes \
        bash scripts/experimental/j63-qwen2-vl-7b-instruct-paligamma-siglip-448/${TASK}.sh /home/jasonlu/workspace/latest/VILA-Internal/runs/train/qwen2-7b-448-baseline-pretrain-20241005211633/model
fi

# pip install lmms-eval==0.2.1
vila-eval \
    --model-path runs/train/$mname/model \
    --model-name $mname \
    --conv-mode $CONV_MODE \
    --tags-include local \
    --report-to wandb

exit 0

bash scripts/experimental/j63-qwen2-vl-7b-instruct-paligamma-siglip-448/sft.sh /home/jasonlu/workspace/latest/VILA-Internal/runs/train/qwen2-7b-448-baseline-pretrain-20241005211633/model
