########################################################
# draco
SLURM_ACCOUNT=${SLURM_ACCOUNT:-"llmservice_nlp_fm"}
SLURM_PARTITION=${SLURM_PARTITION:-"adlr-debug-batch_block4,batch_block1,batch_block2,batch_block3,batch_block4"}

# cs
# SLURM_ACCOUNT=${SLURM_ACCOUNT:-"nvr_elm_llm"}
# SLURM_PARTITION=${SLURM_PARTITION:-"polar4,polar3,polar2,polar,batch_block1,grizzly,,batch_block2,batch_block3"}
########################################################

export BASE_MODEL_PATH=${BASE_MODEL_PATH:-"NousResearch/Llama-2-7b-hf"}
MNAME=$(echo $BASE_MODEL_PATH | rev | cut -d "/" -f 1 | rev)
export VISION_TOWER=${VISION_TOWER:-"google/siglip-large-patch16-384"}
VTOWER=$(echo $VISION_TOWER | rev | cut -d "/" -f 1 | rev)

export ALIGN_DATASET=${1:-llava_1_5_mm_align}
export PT_DATASET=${2:-sharegpt4v_pretrain}
export SFT_DATASET=${3:-sharegpt4v_sft}

echo "$SLURM_ACCOUNT | $SLURM_PARTITION | $MNAME | $VISION_TOWER | $ALIGN_DATASET | $PT_DATASET | $SFT_DATASET"

export BATCH_SIZE=128
export NNODES=4
export ACC_STEP=8

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")
JNAME=$MNAME-$VTOWER-ALIGN-$ALIGN_DATASET-PRETRAIN-$PT_DATASET-SFT-$SFT_DATASET
LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR

ERRF=$LOGDIR/step2-$JNAME.err
LOGF=$LOGDIR/step2-$JNAME.out

OUTPUT_STEP2="/home/yunhaof/workspace/ckpts/vila/data_recipe/llava_align_sharegpt4v_pretrain_sharegpt4v_sft/stage2"
# -pty
# -e $ERRF -o $LOGF \
for i in $(seq 1 4); do

srun -p $SLURM_PARTITION -N $NNODES -t 4:00:00 \
    -A $SLURM_ACCOUNT -J vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash scripts/v1_5/caption/3_sft_captioner.sh $OUTPUT_STEP2 &

done

# slurm_account=llmservice_nlp_fm SLURM_PARTITION=adlr-debug-batch_block4,batch_block1,batch_block2,batch_block3,batch_block4  bash scripts/v1_5/caption/srun_s3.sh llava_1_5_mm_align sharegpt4v_pretrain sharegpt4v_sft+textocr

# squeue --me -o "%.8i %.20P %.100j %.8u %.8T %.8M %.6D %.20S %R"
# export SQUEUE_FORMAT="%.8i %.30P %.120j %.8u %.8T %.8M %.9l %.6D %S %R"
#
# /home/yunhaof/workspace/ckpts/vila/data_recipe/llava_align_sharegpt4v_pretrain_sharegpt4v_sft/stage2
# OUTPUT_STEP2="/home/yunhaof/workspace/ckpts/vila/data_recipe/llava_align_sharegpt4v_pretrain_sharegpt4v_sft/stage2"
# SFT_DATASET=sharegpt4v_sft+hiertext bash scripts/v1_5/caption/3_sft_captioner.sh /home/yunhaof/workspace/ckpts/vila/data_recipe/llava_align_sharegpt4v_pretrain_sharegpt4v_sft/stage2

# slurm_account=nvr_elm_llm bash scripts/v1_5/caption/srun_s3_textocr.sh llava_1_5_mm_align sharegpt4v_pretrain sharegpt4v_sft+hiertext+textocr
#
# bash scripts/v1_5/caption/srun_s3_textocr.sh llava_1_5_mm_align sharegpt4v_pretrain sharegpt4v_sft+hiertext+textocr
# bash scripts/v1_5/caption/srun_s3_textocr.sh llava_1_5_mm_align sharegpt4v_pretrain sharegpt4v_sft+hiertext
# bash scripts/v1_5/caption/srun_s3_textocr.sh llava_1_5_mm_align sharegpt4v_pretrain sharegpt4v_sft+textocr
# bash scripts/v1_5/caption/srun_s3_textocr.sh llava_1_5_mm_align sharegpt4v_pretrain sharegpt4v_sft
