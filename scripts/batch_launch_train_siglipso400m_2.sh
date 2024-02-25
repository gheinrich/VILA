#!/bin/bash
#SBATCH --job-name=valley-pretraining:nvr_lpr_aiagent
#SBATCH --nodes=32
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,grizzly2,polar2
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=pretraining-valley.out


srun --label bash /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/v1_5/video/2_train_mmc4_coyo_sharegpt4v_vicuna_siglipso400m.sh


