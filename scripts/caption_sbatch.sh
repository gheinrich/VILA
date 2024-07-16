#!/bin/bash
#SBATCH -A nvr_elm_llm                  #account
#SBATCH --gres=gpu:8
#SBATCH --partition=grizzly,polar
#SBATCH --exclusive
#SBATCH -t 04:00:00             		#wall time limit, hr:min:sec
#SBATCH -N 1                    		#number of nodes
#SBATCH -J nvr_elm_llm-vlm:label-coyo	#job name
#SBATCH --dependency singleton
#SBATCH --array=0-511%16

WORKDIR=$(pwd)
# TIME=$(date +"%m-%d_%H:%M")
TIME=$(date +"sbatch-%m-%d_%H")
RESULTS="${WORKDIR}/slurm-logs/$TIME"
mkdir -p $RESULTS
# OUTFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT-%j-%n.out"
# ERRFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT-%j-%n.err"
OUTFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT.out"
ERRFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT.err"

# MAX_JOBS=4
# num_nodes=20

srun --label -A nvr_elm_llm -J nvr_elm_llm-vlm:label-coyo-$SLURM_ARRAY_TASK_ID-$SLURM_ARRAY_TASK_COUNT \
    -o $OUTFILE -e $ERRFILE \
    torchrun  --nproc_per_node=8  llava/eval/run_llava_list_coyo.py \
        --model-name ~/downloads/vicuna-13b-clip336-mmc4sub+coyo-finetune-captioner-e4 \
        --conv-mode vicuna_v1_1 \
        --dataset /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/sam-reformat \
        --idx $SLURM_ARRAY_TASK_ID \
        --total $SLURM_ARRAY_TASK_COUNT

# bash ~/workspace/multi-modality-research/VILA/scripts/caption.sh $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
