#/bin/bash
set -e

# export VILA_SLURM_ACCOUNT=llmservice_nlp_fm
export STAGE3_DATASET=${STAGE3_DATASET:-"sharegpt4v_sft+vflan"}

nodes=${1:-"2"}
# fp16, fp8, fp8_memoryefficient
TASK=${TASK:-"fp16"}
mname=$TASK-baseline-nodes_$nodes-$STAGE3_DATASET
CONV_MODE="llama_3"

if [ "$nodes" = "1" ] || [ "$nodes" = "2" ]; then
    VILA_SLURM_PARTITION=$VILA_SLURM_PARTITION,interactive
fi

if [ ! -f runs/train/$mname/model/config.json ]; then
    vila-run -m train -J $mname -N $nodes \
        bash scripts/experimental/te_qlinear/sft_${TASK}.sh
fi

# pip install lmms-eval==0.2.1
vila-eval \
    --model-path runs/train/$mname/model \
    --model-name $mname \
    --conv-mode $CONV_MODE \
    --tags-include local \
    --report-to wandb

exit 0

export VILA_SLURM_ACCOUNT=llmservice_nlp_fm
VILA_SLURM_PARTITION=$VILA_SLURM_PARTITION,interactive vila-run -m train -J fp8-dev -N 1 bash scripts/experimental/te_qlinear/sft_fp8.sh
