#!/bin/bash
#SBATCH --job-name=vila-7b-internvid10m-pretraining:nvr_lpr_aiagent
#SBATCH --nodes=32
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=7b-internvid-10m-training-1.out


srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/1_train_projector_vilavideo7b_v02.sh
# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/3_sft_video_vicuna13b_siglipso400m.sh
# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/3_sft_videov3_vicuna13b_siglipso400m.sh
# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/2_train_mmc4_coyo_sharegpt4v_internvid10m_vicuna13b_siglipso400m_video.sh

