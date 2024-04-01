########################################################
# draco
# SLURM_ACCOUNT=${SLURM_ACCOUNT:-llmservice_nlp_fm}
# SLURM_PARTITION=${SLURM_PARTITION:-batch_block1,batch_block2,batch_block3}

# cs
SLURM_ACCOUNT=${SLURM_ACCOUNT:-"nvr_elm_llm"}
SLURM_PARTITION=${SLURM_PARTITION:-"polar4,polar3,polar2,polar,batch_block1,grizzly,grizzly2,batch_block2,batch_block3"}
########################################################
# bash scripts/v1_5/caption/srun_s2.sh NousResearch/Llama-2-13b-hf llava_1_5_mm_align sharegpt4v_pretrain+coyo_25m+mmc4core
# bash scripts/v1_5/caption/srun_s2.sh NousResearch/Llama-2-13b-hf llava_1_5_mm_align sharegpt4v_pretrain+coyo_25m+mmc4core
# bash scripts/v1_5/caption/srun_s2.sh NousResearch/Llama-2-13b-hf llava_1_5_mm_align sharegpt4v_pretrain

# export DATASET=${DATASET:-sharegpt4v_pretrain+coyo_25m_wds}
# PT_DATASET
export VISION_TOWER=${VISION_TOWER:-"google/siglip-large-patch16-384"}
export BASE_MODEL_PATH=${1:-"NousResearch/Llama-2-7b-hf"}
export ALIGN_DATASET=${2:-llava_1_5_mm_align}
export PT_DATASET=${3:-sharegpt4v_pretrain}

MNAME=$(echo $BASE_MODEL_PATH | rev | cut -d "/" -f 1 | rev)
VTOWER=$(echo $VISION_TOWER | rev | cut -d "/" -f 1 | rev)

echo "$SLURM_ACCOUNT | $SLURM_PARTITION | $ALIGN_DATASET | $PT_DATASET"

export BATCH_SIZE=128
export NNODES=8
export ACC_STEP=4

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")
JNAME=$MNAME-$VTOWER-ALIGN-$ALIGN_DATASET-PRETRAIN-$PT_DATASET
LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR

ERRF=$LOGDIR/step2-$JNAME.err 
LOGF=$LOGDIR/step2-$JNAME.out

# -pty
# -e $ERRF -o $LOGF \
for i in $(seq 1 8); do 

srun -p $SLURM_PARTITION -N $NNODES -t 4:00:00 \
    -A $SLURM_ACCOUNT -J vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash scripts/v1_5/caption/2_pretrain.sh &

done
# bash scripts/v1_5/caption/srun_s2.sh llava_1_5_mm_align sharegpt4v_pretrain
# bash scripts/v1_5/caption/srun_s2.sh llava_1_5_mm_align sharegpt4v_pretrain+coyo_25m_wds+mmc4core
# bash scripts/v1_5/caption/srun_s2.sh llava_1_5_mm_align sharegpt4v_pretrain+coyo_25m_wds+mmc4core