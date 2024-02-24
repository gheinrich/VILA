#!/bin/bash
#SBATCH --job-name=nvr_lpr_aiagent-vlm:pretraining-siglipso400m-videov3
#SBATCH --nodes=32
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=pretraining-videov3.out


srun --label bash /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/vicuna/2_sft_videov3_siglipso400m.sh

