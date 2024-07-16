#!/bin/bash
#SBATCH --job-name=vila-7b-internvid10m-pretraining:nvr_lpr_aiagent
#SBATCH --nodes=16
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,polar2,polar3,polar4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=7b-internvid-10m-ego1m-training.out


# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/1_train_projector_vicuna13b_siglipso400m.sh
srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/1_mm_align.sh
# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/3_sft_videov2_siglipso400m_2.sh
