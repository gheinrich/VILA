########################################################
# draco
# SLURM_ACCOUNT=${SLURM_ACCOUNT:-llmservice_nlp_fm}
# SLURM_PARTITION=${SLURM_PARTITION:-batch_block1,batch_block2,batch_block3}

# cs
SLURM_ACCOUNT=${SLURM_ACCOUNT:-"nvr_elm_llm"}
SLURM_ACCOUNT=nvr_elm_llm
SLURM_PARTITION=${SLURM_PARTITION:-"polar3,polar2,polar,batch_block1,grizzly,,batch_block2,batch_block3"}
########################################################
export VISION_TOWER=${VISION_TOWER:-"OpenGVLab/InternViT-6B-448px-V1-2"}
export BASE_MODEL_PATH=${1:-"NousResearch/Nous-Hermes-2-Yi-34B"}
export ALIGN_DATASET=${2:-llava_1_5_mm_align}

MNAME=$(echo $BASE_MODEL_PATH | rev | cut -d "/" -f 1 | rev)
VTOWER=$(echo $VISION_TOWER | rev | cut -d "/" -f 1 | rev)

echo "$SLURM_ACCOUNT | $SLURM_PARTITION | $ALIGN_DATASET | $BASE_MODEL_PATH | $VISION_TOWER"

export BATCH_SIZE=32
export NNODES=8
export ACC_STEP=1

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")
# JNAME=$MNAME-$VTOWER-align-$ALIGN_DATASET
JNAME=vila-40b-v1.5-mm-align

LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR

ERRF=$LOGDIR/step1-$JNAME.err
LOGF=$LOGDIR/step1-$JNAME.out

for i in $(seq 1 8); do

srun -p $SLURM_PARTITION -N $NNODES -t 4:00:00 \
    -A $SLURM_ACCOUNT -J vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash 40b/1_mm_align.sh $BASE_MODEL_PATH ./checkpoints/$JNAME &

done
