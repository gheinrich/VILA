########################################################
# draco
# slurm_account=${slurm_account:-llmservice_nlp_fm}
# slurm_partition=${slurm_partition:-batch_block1,batch_block2,batch_block3}

# cs
slurm_account=${slurm_account:-"nvr_elm_llm"}
slurm_partition=${slurm_partition:-"polar3,polar2,polar,batch_block1,grizzly,grizzly2,batch_block2,batch_block3"}
########################################################
export ALIGN_DATASET=${1:-llava_1_5_mm_align}
# export DATASET=${DATASET:-vflan_llava_1_5_sft}

echo "$slurm_account | $slurm_partition"

# export BATCH_SIZE=128
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
srun -p $slurm_partition -N $NNODES -t 4:00:00 \
    -A $slurm_account -J vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --exclude batch-block7-00105,batch-block7-00113,batch-block7-00123,batch-block7-00139 \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash scripts/v1_5/caption/1_mm_align.sh &

# bash scripts/v1_5/captioner/srun_s1.sh llava_1_5_mm_align
# bash scripts/v1_5/captioner/srun_s1.sh ccs_recap_wds
# bash scripts/v1_5/captioner/srun_s1.sh llava_1_5_mm_align+ccs_recap_wds