########################################################
# draco
# SLURM_ACCOUNT=${SLURM_ACCOUNT:-llmservice_nlp_fm}
# SLURM_PARTITION=${SLURM_PARTITION:-batch_block1,batch_block2,batch_block3}

# cs
SLURM_ACCOUNT=${SLURM_ACCOUNT:-"nvr_elm_llm"}
SLURM_PARTITION=${SLURM_PARTITION:-"polar3,polar2,polar,batch_block1,grizzly,grizzly2,batch_block2,batch_block3"}
########################################################
export VISION_TOWER=${VISION_TOWER:-"google/siglip-large-patch16-384"}
export BASE_MODEL_PATH=${1:-"NousResearch/Llama-2-7b-hf"}
export ALIGN_DATASET=${2:-llava_1_5_mm_align}

MNAME=$(echo $BASE_MODEL_PATH | rev | cut -d "/" -f 1 | rev)
VTOWER=$(echo $VISION_TOWER | rev | cut -d "/" -f 1 | rev)

echo "$SLURM_ACCOUNT | $SLURM_PARTITION | $ALIGN_DATASET | $BASE_MODEL_PATH"

# export BATCH_SIZE=128
export NNODES=4
export ACC_STEP=8

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")
JNAME=$MNAME-$VTOWER-align-$ALIGN_DATASET
LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR

ERRF=$LOGDIR/step1-$JNAME.err 
LOGF=$LOGDIR/step1-$JNAME.out

# -pty
# -e $ERRF -o $LOGF \
srun -p $SLURM_PARTITION -N $NNODES -t 4:00:00 \
    -A $SLURM_ACCOUNT -J vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash scripts/v1_5/caption/1_mm_align.sh &

# bash scripts/v1_5/captioner/srun_s1.sh NousResearch/Llama-2-7b-hf llava_1_5_mm_align
# bash scripts/v1_5/captioner/srun_s1.sh ccs_recap_wds
# bash scripts/v1_5/captioner/srun_s1.sh llava_1_5_mm_align+ccs_recap_wds
# SLURM_ACCOUNT=llmservice_nlp_fm SLURM_PARTITION=adlr-debug-batch_block4,batch_block1,batch_block2,batch_block3,batch_block4 \
#     BASE_MODEL_PATH=NousResearch/Llama-2-13b-hf \
#     bash scripts/v1_5/caption/srun_s1.sh NousResearch/Llama-2-7b-hf llava_1_5_mm_align