#/bin/bash
set -e

# export VILA_SLURM_ACCOUNT=llmservice_nlp_fm
nodes=${1:-"2"}

MODEL=${MODEL:-"qwen2-vl-7b-instruct"}

# 1_align
# 2_pretrain
# 3_sft
TASK=${TASK:-"3_sft"}
# mname=${MODEL}_${TASK}_${nodes}

export DATASETS=${2:-"sharegpt4v_sft"}
JOB_NAME=${MODEL}_J65-${TASK}-${DATASETS}-nodes_${nodes}

if [ "$nodes" = "1" ] || [ "$nodes" = "2" ]; then
    # for debugging purpose
    VILA_SLURM_PARTITION=$VILA_SLURM_PARTITION,interactive
fi

if [ ! -f runs/train/$JOB_NAME/model/config.json ]; then
    vila-run -m train -J $JOB_NAME -N $nodes \
        bash scripts/v2_0/qwen2-vl-7b-instruct/${TASK}.sh /home/jasonlu/workspace/latest/VILA-Internal/runs/train/qwen2-7b-448-baseline-pretrain-20241005211633/model
fi

# exit 0

# evaluation
echo "launch evaluation for $JOB_NAME"
CONV_MODE="auto"
export WANDB_PROJECT="vila-data-curation"

vila-eval \
    --model-path runs/train/$JOB_NAME/model \
    --model-name $JOB_NAME \
    --conv-mode $CONV_MODE \
    --tags-include core \
    --report-to wandb
