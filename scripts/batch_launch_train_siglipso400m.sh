#!/bin/bash
#SBATCH --job-name=internvid-sft:nvr_lpr_aiagent
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --time=1:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,grizzly2,polar2
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=internvid-sft-test.out


srun --label bash /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/v1_5/video/3_sft_videov2_siglipso400m_test.sh
# srun --label bash /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/v1_5/video/2_train_mmc4_coyo_sharegpt4v_internvid1m_vicuna_siglipso400m.sh


