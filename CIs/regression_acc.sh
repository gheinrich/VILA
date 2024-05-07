export SLURM_ACCOUNT=nvr_elm_llm
# dtime=$(TZ=Asia/Shanghai date +"%b_%d")
suffix=${1:-"42"}

export SEED=$suffix

_VILA_CI_RECIPIENTS=ligengz@nvidia.com,jasonlu@nvidia.com,yunhaof@nvidia.com,fuzhaox@nvidia.com

dtime=$(date +"%b_%d")
WORKDIR=./checkpoints/vila-regression/$dtime-$suffix
CKPT1=$WORKDIR/CIs-reproduce-stage1
CKPT2=$WORKDIR/CIs-reproduce-stage2
CKPT3=$WORKDIR/CIs-reproduce-stage3-image
CKPT3_video=$WORKDIR/CIs-reproduce-stage3-video

# rm $CKPT1 $CKPT2 $CKPT3
wait

echo "[$SLURM_ACCOUNT] working directory $WORKDIR"
mkdir -p slurm-logs/regression

# ALIGN_DATASET=llava_1_5_mm_align bash scripts/reproduce/1_mm_align.sh $CKPT1
# PT_DATASET=sharegpt4v_pretrain bash scripts/reproduce/2_pretrain.sh $CKPT1 $CKPT2
# SFT_DATASET=sharegpt4v_sft bash scripts/reproduce/3_sft_captioner.sh $CKPT2 $CKPT3

export NNODES=2
export ACC_STEP=16

jname=stage1-$suffix
srun -A $SLURM_ACCOUNT \
    -N 1 \
    -p $SLURM_PARTITION -t 4:00:00 \
    -J vila:CI-acc-regression-$suffix \
    --dependency singleton \
    -e slurm-logs/regression-$dtime-$suffix/eval-$jname.err \
    -o slurm-logs/regression-$dtime-$suffix/eval-$jname.out \
    --gpus-per-node 8 --exclusive \
    bash scripts/reproduce/1_mm_align.sh llava_1_5_mm_align $CKPT1 &

jname=stage2-$suffix
srun -A $SLURM_ACCOUNT \
    -N 1 \
    -p $SLURM_PARTITION -t 4:00:00 \
    -J vila:CI-acc-regression-$suffix \
    --dependency singleton \
    -e slurm-logs/regression-$dtime-$suffix/eval-$jname.err \
    -o slurm-logs/regression-$dtime-$suffix/eval-$jname.out \
    --gpus-per-node 8 --exclusive \
    bash scripts/reproduce/2_pretrain.sh sharegpt4v_pretrain $CKPT1 $CKPT2 &

# Image SFT
jname=stage3-$suffix
srun -A $SLURM_ACCOUNT \
    -N 1 \
    -p $SLURM_PARTITION -t 4:00:00 \
    -J vila:CI-acc-regression-$suffix \
    --dependency singleton \
    -e slurm-logs/regression-$dtime-$suffix/eval-$jname.err \
    -o slurm-logs/regression-$dtime-$suffix/eval-$jname.out \
    --gpus-per-node 8 --exclusive \
    bash scripts/reproduce/3_sft_captioner.sh sharegpt4v_sft $CKPT2 $CKPT3 &

# Video SFT
jname=stage3-$suffix
# for i in $(seq 1 5); do
# srun -A $SLURM_ACCOUNT \
#     -N 16 \
#     -p $SLURM_PARTITION -t 4:00:00 \
#     -J vila:CI-acc-regression-$suffix \
#     --dependency singleton \
#     --gpus-per-node 16 --exclusive \
#     SFT_DATASET=vflan+sharegpt4v_sft+video_chatgpt+shot2story_shotonly bash scripts/reproduce/3_sft_captioner.sh $CKPT2 $CKPT3 &
# done
# wait 
wait 

exit 0
bash scripts/v1_5/caption/slurm_eval_all.sh $CKPT3 $WORKDIR/eval_output
wait 

python CIs/send_email.py \
    --title="[VILA] Regression Test Report" \
    --recipients $_VILA_CI_RECIPIENTS \
    --pure_text "Regression Test Finish at $WORKDIR/eval_output"
