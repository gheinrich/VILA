#!/bin/bash
#SBATCH -A llmservice_nlp_fm                  #account
#SBATCH --partition=cpu,cpu_1,cpu_long
#SBATCH --exclusive
#SBATCH -t 02:30:00             		#wall time limit, hr:min:sec
#SBATCH -N 1                    		#number of nodes
#SBATCH -J llmservice_nlp_fm-vlm:caption-sharding	#job name
#SBATCH --dependency singleton
#SBATCH --array=0-63

WORKDIR=$(pwd)
# TIME=$(date +"%m-%d_%H:%M")
TIME=$(date +"%m-%d_%H")
RESULTS="${WORKDIR}/slurm-logs/$TIME"
mkdir -p $RESULTS
OUTFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT-%j-%n.out"
ERRFILE="${RESULTS}/slurm-$SLURM_ARRAY_TASK_ID-of-$SLURM_ARRAY_TASK_COUNT-%j-%n.err"

# MAX_JOBS=4
# num_nodes=20

srun --label -N 1 -t 4:00:00 -J llmservice_nlp_fm-vlm:caption-sharding-$SLURM_ARRAY_TASK_ID-$SLURM_ARRAY_TASK_COUNT \
    -o $OUTFILE -e $ERRFILE \
    python shard_caption.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
