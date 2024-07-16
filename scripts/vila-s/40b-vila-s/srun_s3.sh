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
export PRETRAIN_DATASET=${2:-coyo_25m+mmc4core+sharegpt4v_pretrain}
export SFT_DATASET=${2:-OSS_mixture}

MNAME=$(echo $BASE_MODEL_PATH | rev | cut -d "/" -f 1 | rev)
VTOWER=$(echo $VISION_TOWER | rev | cut -d "/" -f 1 | rev)

echo "$SLURM_ACCOUNT | $SLURM_PARTITION | $PRETRAIN_DATASET | $BASE_MODEL_PATH | $VISION_TOWER"

export BATCH_SIZE=32
export NNODES=16
export ACC_STEP=1

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")
JNAME_PREV=VILA1.5-34b-stage2
JNAME=$JNAME_PREV-SFT-$SFT_DATASET

LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR

ERRF=$LOGDIR/step2-$JNAME.err
LOGF=$LOGDIR/step2-$JNAME.out

for i in $(seq 1 10); do

srun -p $SLURM_PARTITION -N $NNODES -t 4:00:00 \
    -A $SLURM_ACCOUNT -J vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash 40b/3_sft.sh Efficient-Large-Model/VILA1.5-34b-stage2 ./checkpoints/$JNAME &

done
