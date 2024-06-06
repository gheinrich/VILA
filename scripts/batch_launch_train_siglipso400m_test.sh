#!/bin/bash
#SBATCH --job-name=ego4d-pretraining:nvr_lpr_aiagent
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --time=1:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=interactive,grizzly,polar,polar2,polar3,polar4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=training-test-3-01.out


srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/1_mm_align_test.sh
# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/1_train_projector_vicuna13b_siglipso400m_video_test.sh


