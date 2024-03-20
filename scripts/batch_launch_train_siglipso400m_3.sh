#!/bin/bash
#SBATCH --job-name=vila-init-7b-pretraining:nvr_lpr_aiagent
#SBATCH --nodes=32
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=13b-internvid-1m-training-2.out


srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/1_train_projector_vicuna_siglipso400m_video.sh
# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/3_sft_videov2_siglipso400m_2.sh
