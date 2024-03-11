# bash scripts/v1_5/captioner/srun_s2.sh llava_1_5_mm_align sharegpt4v_pretrain+coyo_25m_wds
# bash scripts/v1_5/captioner/srun_s2.sh llava_1_5_mm_align sharegpt4v_pretrain+coyo_25m_wds_recap
#
export ALIGN_DATASET=${1:-llava_1_5_mm_align}
export PT_DATASET=${2:-sharegpt4v_pretrain}
# export DATASET=${DATASET:-vflan_llava_1_5_sft}

export BATCH_SIZE=128
export NNODES=8
export ACC_STEP=4
# export PARTITION=${PARTITION:-llmservice_nlp_fm}
# PARTITION=nvr_elm_llm
export PARTITION=${PARTITION:-llmservice_nlp_fm}

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")
JNAME=ALIGN-$ALIGN_DATASET-PRETRAIN-$PT_DATASET
LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR


ERRF=$LOGDIR/step2-$JNAME.err 
LOGF=$LOGDIR/step2-$JNAME.out

# -pty
# -e $ERRF -o $LOGF \
for i in $(seq 1 10); do 

srun -p batch_block1,batch_block2,batch_block3 -N $NNODES -t 4:00:00 \
    -A $PARTITION -J vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash scripts/v1_5/captioner/2_pretrain.sh &

done
# bash scripts/v1_5/captioner/srun_s2.sh llava_1_5_mm_align sharegpt4v_pretrain