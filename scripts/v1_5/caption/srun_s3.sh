########################################################
# draco
# slurm_account=${slurm_account:-llmservice_nlp_fm}
# slurm_partition=${slurm_partition:-batch_block1,batch_block2,batch_block3}

# cs
slurm_account=${slurm_account:-"nvr_elm_llm"}
slurm_partition=${slurm_partition:-"polar4,polar3,polar2,polar,batch_block1,grizzly,grizzly2,batch_block2,batch_block3"}
########################################################
# bash scripts/v1_5/caption/srun_s3.sh llava_1_5_mm_align sharegpt4v_pretrain+coyo_25m_wds
# bash scripts/v1_5/caption/srun_s3.sh llava_1_5_mm_align sharegpt4v_pretrain+coyo_25m_wds_recap


export ALIGN_DATASET=${1:-llava_1_5_mm_align}
export PT_DATASET=${2:-sharegpt4v_pretrain}
export SFT_DATASET=${3:-sharegpt4v_sft}

echo "$slurm_account | $slurm_partition | $ALIGN_DATASET | $PT_DATASET | $SFT_DATASET"


export BATCH_SIZE=128
export NNODES=4
export ACC_STEP=8
# export PARTITION=${PARTITION:-llmservice_nlp_fm}
# PARTITION=nvr_elm_llm
export PARTITION=${PARTITION:-llmservice_nlp_fm}

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")
JNAME=ALIGN-$ALIGN_DATASET-PRETRAIN-$PT_DATASET-SFT-$SFT_DATASET
LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR


ERRF=$LOGDIR/step2-$JNAME.err 
LOGF=$LOGDIR/step2-$JNAME.out

# -pty
# -e $ERRF -o $LOGF \
for i in $(seq 1 4); do 

srun -p $slurm_partition -N $NNODES -t 4:00:00 \
    -A $slurm_account -J vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash scripts/v1_5/caption/3_sft_captioner.sh &

done
# slurm_account=llmservice_nlp_fm slurm_partition=batch_block1,batch_block2,batch_block3 bash scripts/v1_5/caption/srun_s3.sh llava_1_5_mm_align sharegpt4v_pretrain+coyo_25m_wds+mmc4core
# bash scripts/v1_5/caption/srun_s3.sh llava_1_5_mm_align sharegpt4v_pretrain sharegpt4v_sft+vflan
# slurm_account=llmservice_nlp_fm slurm_partition=adlr-debug-batch_block4,batch_block1,batch_block2,batch_block3,batch_block4  bash scripts/v1_5/caption/srun_s3.sh llava_1_5_mm_align sharegpt4v_pretrain sharegpt4v_sft+vflan

# squeue --me -o "%.8i %.20P %.100j %.8u %.8T %.8M %.6D %.20S %R"
# export SQUEUE_FORMAT="%.8i %.30P %.120j %.8u %.8T %.8M %.9l %.6D %S %R"