########################################################
# draco
SLURM_ACCOUNT=${SLURM_ACCOUNT:-"llmservice_nlp_fm"}
SLURM_PARTITION=${SLURM_PARTITION:-"adlr-debug-batch_block4,batch_block1,batch_block2,batch_block3,batch_block4"}

# cs
# SLURM_ACCOUNT=${SLURM_ACCOUNT:-"nvr_elm_llm"}
# SLURM_PARTITION=${SLURM_PARTITION:-"polar4,polar3,polar2,polar,batch_block1,grizzly,grizzly2,batch_block2,batch_block3"}
########################################################

export BASE_MODEL_PATH=${BASE_MODEL_PATH:-"lmsys/vicuna-13b-v1.5"}
MNAME=$(echo $BASE_MODEL_PATH | rev | cut -d "/" -f 1 | rev)
export VISION_TOWER=${VISION_TOWER:-"openai/clip-vit-large-patch14-336"}
VTOWER=$(echo $VISION_TOWER | rev | cut -d "/" -f 1 | rev)

export ALIGN_DATASET=${1:-llava_1_5_mm_align}
export PT_DATASET=${2:-sharegpt4v_pretrain}
export SFT_DATASET=${3:-sharegpt4v_sft}

echo "$SLURM_ACCOUNT | $SLURM_PARTITION | $MNAME | $VISION_TOWER | $ALIGN_DATASET | $PT_DATASET | $SFT_DATASET"

export BATCH_SIZE=128
export NNODES=8
export ACC_STEP=4

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")
JNAME=$MNAME-$VTOWER-ALIGN-$ALIGN_DATASET-PRETRAIN-$PT_DATASET-SFT-$SFT_DATASET
LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR

ERRF=$LOGDIR/step2-$JNAME.err 
LOGF=$LOGDIR/step2-$JNAME.out

OUTPUT_STEP2="/home/ligengz/workspace/video_checkpoint/video-13b"
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
# bash scripts/v1_5/caption/srun_s3_video.sh llava_1_5_mm_align sharegpt4v_pretrain panda70m
# bash scripts/v1_5/caption/srun_s3_video.sh llava_1_5_mm_align sharegpt4v_pretrain shot2story_shotonly
# bash scripts/v1_5/caption/srun_s3_video.sh llava_1_5_mm_align sharegpt4v_pretrain panda70m+shot2story_shotonly