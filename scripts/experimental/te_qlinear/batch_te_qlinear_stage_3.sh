#!/bin/bash
#SBATCH --job-name=reproduce_vila:stage2
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_elm_llm
#SBATCH --partition=grizzly,polar,polar2,polar3,polar4
#SBATCH --exclusive

CKPT2=checkpoints/CIs-reproduce-stage2
CKPT3=checkpoints/CIs-qlinear-stage3
SFT_DATASET=sharegpt4v_sft

LUSTRE_DIR=$HOME
TIME=$(TZ=Asia/Shanghai date +"%m-%d_%H:%M")
RESULTS="${LUSTRE_DIR}/slurm-logs/$TIME-%j"
OUTFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT-%j-%n.out"
ERRFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT-%j-%n.err"

srun \
    -o $OUTFILE -e $ERRFILE \
    bash scripts/te_qlinear/3_sft_captioner.sh $SFT_DATASET $CKPT2 $CKPT3
