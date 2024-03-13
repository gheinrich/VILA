#!/bin/bash
#SBATCH --job-name=internvid-pretraining:nvr_lpr_aiagent
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=video-training-test.out


srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/1_train_projector_vicuna_siglipso400m_test.sh
# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/2_train_mmc4_coyo_sharegpt4v_internvid1m_vicuna_siglipso400m.sh

