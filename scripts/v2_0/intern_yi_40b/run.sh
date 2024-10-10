#/bin/bash
set -e

# export VILA_SLURM_ACCOUNT=llmservice_nlp_fm
nodes=${1:-"2"}

# 1_align
# 2_pretrain
# 3_sft
TASK=${TASK:-"2_pretrain"}
mname=$TASK-yi34b-nodes_$nodes
CONV_MODE="llama_3"

if [ "$nodes" = "1" ] || [ "$nodes" = "2" ]; then
    # for debugging purpose
    VILA_SLURM_PARTITION=$VILA_SLURM_PARTITION,interactive
fi

if [ ! -f runs/train/$mname/model/config.json ]; then
    vila-run -m train -J $mname -N $nodes --max-retry 3 \
        bash scripts/v2_0/intern_yi_40b/${TASK}.sh
fi

exit 0

# evaluation
CONV_MODE="hermes-2"
vila-eval \
    --model-path runs/train/$mname/model \
    --model-name $mname \
    --conv-mode $CONV_MODE \
    --tags-include local \
    --report-to wandb
