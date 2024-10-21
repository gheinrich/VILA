#/bin/bash
set -e

# Qwen2-VL-7B-Instruct
MODEL=${MODEL:-"qwen2-vl-7b-instruct"}
BASEDIR=runs/train

mkdir -p $BASEDIR
rm -f core.*

# Handle interrupt signal and cleanup all slurm (training) jobs
exitfn () {
    trap SIGINT  # Restore signal handling for SIGINT.
    for stage in align pretrain sft; do
        jname=$VILA_SLURM_ACCOUNT:train/$MODEL-$stage
        echo 'Aarghh Interuption! Canceling jobs' $jname
        scancel -n $jname
    done
    exit -1
}
trap "exitfn" INT


# if [ ! -f $BASEDIR/$MODEL-align/model/config.json ]; then
#     vila-run \
#         --mode train \
#         --job-name $MODEL-align \
#         --nodes 8 \
#         --output-dir $BASEDIR/$MODEL-align \
#         scripts/v2_0/$MODEL/1_align.sh
# fi

# if [ ! -f $BASEDIR/$MODEL-pretrain/model/config.json ]; then
#     vila-run \
#         --mode train \
#         --job-name $MODEL-pretrain \
#         --nodes 16 \
#         --output-dir $BASEDIR/$MODEL-pretrain \
#         scripts/v2_0/$MODEL/2_pretrain.sh \
#             $BASEDIR/$MODEL-align/model
# fi

huggingface-cli download Efficient-Large-Model/qwen2-vl-7b-instruct-pretrain
export DATASETS=${1:-"sharegpt4v_sft"}
JOB_NAME=$MODEL-sft_$DATASETS

if [ ! -f $BASEDIR/$JOB_NAME/model/config.json ]; then
    vila-run \
        --mode train \
        --job-name $JOB_NAME \
        --nodes 16 \
        --output-dir $BASEDIR/$JOB_NAME \
        scripts/v2_0/$MODEL/3_sft.sh \
            $BASEDIR/$MODEL-pretrain/model Efficient-Large-Model/qwen2-vl-7b-instruct-pretrain
fi

echo "Training finished, now launch evaluation"

# TODO(zhijian, ligeng): pin this to pyproject.toml.
# pip install lmms-eval==0.2.1

# add interactive partition to accelerate
# export VILA_SLURM_PARTITION=$VILA_SLURM_PARTITION,interactive
# export WANDB_ENTITY="efficient-large-model"
export WANDB_PROJECT="vila-eval-qwen2"
CONV_MODE="auto"

vila-eval \
    --model-path $BASEDIR/$JOB_NAME/model \
    --model-name $JOB_NAME \
    --conv-mode $CONV_MODE \
    --tags-include regression \
    --output-dir $BASEDIR/$JOB_NAME-eval \
    --report-to wandb
