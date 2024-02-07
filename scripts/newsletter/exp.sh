#export DATASET=
#   datacomp_webds+mmc4core+sharegpt4v_pretrain
#   datacomp_webds+coyo_webds_vila+mmc4core
#   coyo_25m_refilter+mmc4core
#   coyo_25m_recap+mmc4core
#   coyo_webds_vila+mmc4core

# PT_DATASET=datacomp_webds+mmc4core bash scripts/newsletter/exp.sh
export PT_DATASET=${PT_DATASET:-coyo_25m_recap+mmc4core}
export DATASET=${DATASET:-vflan_llava_1_5_sft}
export NNODES=8
export ACC_STEP=8

JNAME=PT-$PT_DATASET-SFT-$DATASET

dtime=$(TZ=Asia/Shanghai date +"%b_%d-%H")
LOGDIR=slurm-logs/$dtime
mkdir -p $LOGDIR
# JNAME=step1-PT-$PT_DATASET
ERRF=$LOGDIR/step1-$JNAME.err 
LOGF=$LOGDIR/step1-$JNAME.out

echo "launch $JNAME"

for i in $(seq 1 10); do 
srun -p batch_block1,batch_block2,batch_block3 -A llmservice_nlp_fm \
    -N $NNODES -t 4:00:00 -J llmservice_nlp_fm-vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash scripts/newsletter/1_train_mmc4_coyo_vicuna_64_refilter.sh &
done

ERRF=$LOGDIR/step2-$JNAME.err 
LOGF=$LOGDIR/step2-$JNAME.out
for i in $(seq 1 3); do 
# srun -p grizzly,polar -A nvr_elm_llm -N 32 -t 4:00:00 -J nvr_elm_llm-vila:$DATASET \
srun -p batch_block1,batch_block2,batch_block3 -A llmservice_nlp_fm \
    -N $NNODES -t 4:00:00 -J llmservice_nlp_fm-vila:$JNAME \
    --gpus-per-node 8 --exclusive \
    --dependency singleton \
    -e $ERRF -o $LOGF \
    bash scripts/newsletter/2_sft_vlan_sft.sh &
done