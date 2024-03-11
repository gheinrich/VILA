export ALIGN_DATASET=${1:-llava_1_5_mm_align}
# export DATASET=${DATASET:-vflan_llava_1_5_sft}

export BATCH_SIZE=128
export NNODES=4
export ACC_STEP=8

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")
JNAME=ALIGN-$ALIGN_DATASET
LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR


ERRF=$LOGDIR/step1-$JNAME.err 
LOGF=$LOGDIR/step1-$JNAME.out

# -pty
# -e $ERRF -o $LOGF \
srun -p batch_block1,batch_block2,batch_block3 -N $NNODES -t 4:00:00 \
    -A llmservice_nlp_fm -J llmservice_nlp_fm-vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash scripts/v1_5/captioner/1_mm_align.sh &

# bash scripts/v1_5/captioner/srun_s1.sh llava_1_5_mm_align
# bash scripts/v1_5/captioner/srun_s1.sh ccs_recap_wds
# bash scripts/v1_5/captioner/srun_s1.sh llava_1_5_mm_align+ccs_recap_wds