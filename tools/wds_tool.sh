#!/bin/bash
#SBATCH --account=nvr_elm_llm
#SBATCH --partition=cpu,cpu_long,cpu_short,cpu_interactive,interactive
#SBATCH --job-name=nvr_elm_llm:wds
#SBATCH --output=runs/wds/gen-%A_%a.out
#SBATCH --error=runs/wds/gen-%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --array=0-100:10  # total tars to transfer

set -e

idx=${SLURM_ARRAY_TASK_ID}
steps=10

DATADIR="/lustre/fs11/portfolios/nvr/projects/nvr_elm_llm/dataset/vila-sft/pixmo"
TARDIR="/home/ligengz/nvr_elm_llm/dataset/vila-sft-tar"
REMOTEDIR=login-eos:$TARDIR


python tools/create_wds.py --folder $DATADIR --start $idx --output_folder $TARDIR/$idx --seg $steps
rsync -avP $TARDIR/$idx/ $REMOTEDIR
rm -rfv $TARDIR/$idx

exit 0
