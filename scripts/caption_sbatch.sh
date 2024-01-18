#!/bin/bash
#SBATCH -A nvr_elm_llm                  #account
#SBATCH --gres=gpu:8
#SBATCH --partition=batch_block1
#SBATCH --exclusive
#SBATCH -t 04:00:00             		#wall time limit, hr:min:sec
#SBATCH -N 1                    		#number of nodes
#SBATCH -J nvr_elm_llm-vlm:label-coyo	#job name
#SBATCH --dependency singleton
#SBATCH --array=0-511

WORKDIR=$(pwd)
# TIME=$(date +"%m-%d_%H:%M")
TIME=$(date +"%m-%d_%H")
RESULTS="${WORKDIR}/slurm-logs/$TIME"
mkdir -p $RESULTS
OUTFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT-%j-%n.out"
ERRFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT-%j-%n.err"

# MAX_JOBS=4
# num_nodes=20

srun --label -N 1 -t 4:00:00 -J nvr_elm_llm-vlm:label-coyo-$SLURM_ARRAY_TASK_ID-$SLURM_ARRAY_TASK_COUNT \
    -o $OUTFILE -e $ERRFILE \
    bash ~/workspace/multi-modality-research/VILA/scripts/caption.sh $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT

 
