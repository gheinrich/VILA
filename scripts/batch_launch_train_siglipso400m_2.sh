#!/bin/bash
#SBATCH --job-name=internvid-pretraining:nvr_lpr_aiagent
#SBATCH --nodes=32
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,grizzly2,polar2,polar3
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=pretraining-internvid1m.out


srun --label bash /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/v1_5/video/3_sft_videov2_siglipso400m_2.sh

