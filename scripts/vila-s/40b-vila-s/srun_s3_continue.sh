########################################################
# draco
# SLURM_ACCOUNT=${SLURM_ACCOUNT:-llmservice_nlp_fm}
# SLURM_PARTITION=${SLURM_PARTITION:-batch_block1,batch_block2,batch_block3}

# cs
SLURM_ACCOUNT=${SLURM_ACCOUNT:-"nvr_elm_llm"}
SLURM_PARTITION=${SLURM_PARTITION:-"polar3,polar2,polar,batch_block1,grizzly,,batch_block2,batch_block3"}
########################################################
export VISION_TOWER=${VISION_TOWER:-"OpenGVLab/InternViT-6B-448px-V1-2"}
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-"NousResearch/Nous-Hermes-2-Yi-34B"}
export PRETRAIN_DATASET=${PRETRAIN_DATASET:-coyo_25m+mmc4core+sharegpt4v_pretrain}

MNAME=$(echo $BASE_MODEL_PATH | rev | cut -d "/" -f 1 | rev)
VTOWER=$(echo $VISION_TOWER | rev | cut -d "/" -f 1 | rev)

export BATCH_SIZE=32
export NNODES=16
export ACC_STEP=1

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")

JNAME_PREV=${1:-"checkpoints/vila-40b-oss-stage2-PRETRAIN-coyo25m_0to5_vila15_40b_recap"}
export SFT_DATASET=${2:-OSS_mixture}
JNAME=$JNAME_PREV-SFT-$SFT_DATASET

echo "$SLURM_ACCOUNT | $SLURM_PARTITION | $PRETRAIN_DATASET | $JNAME_PREV | $VISION_TOWER"

LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR

ERRF=$LOGDIR/step3-$JNAME.err 
LOGF=$LOGDIR/step3-$JNAME.out

for i in $(seq 1 10); do 

srun -p $SLURM_PARTITION -N $NNODES -t 4:00:00 \
    -A $SLURM_ACCOUNT -J vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash 40b/3_sft.sh $JNAME_PREV $JNAME &

done
'''
SLURM_ACCOUNT=nvr_elm_llm 
 cosmos_misc
SLURM_ACCOUNT=cosmos_misc bash 40b/srun_s3_continue.sh checkpoints/vila-40b-oss-stage2-PRETRAIN-sam_0to5_vila40b_recap
SLURM_ACCOUNT=nvr_elm_llm bash 40b/srun_s3_continue.sh checkpoints/vila-40b-oss-stage2-PRETRAIN-coyo25m_0to5_vila15_40b_recap+sam_0to5_vila40b_recap
SLURM_ACCOUNT=nvr_elm_llm bash 40b/srun_s3_continue.sh checkpoints/vila-40b-oss-stage2-PRETRAIN-coyo25m_0to5_vila15_40b_recap
'''