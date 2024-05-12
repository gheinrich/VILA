#!/bin/bash
#SBATCH --job-name=vila-7b-internvid10m-pretraining:nvr_lpr_aiagent
#SBATCH --nodes=32
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
<<<<<<< HEAD
#SBATCH --partition=grizzly,polar,polar2,polar3,polar4
=======
#SBATCH --partition=grizzly,polar,,polar2,polar3,polar4
>>>>>>> 4f906a9425c2318c89084b2a7eda650bc32f05ac
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=7b-internvid-10m-ego1m-training.out


# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/1_train_projector_vicuna13b_siglipso400m.sh
srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/2_train_mmc4_coyo_sharegpt4v_internvid10m_ego4d1m_vicuna_siglipso400m_video.sh
# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/3_sft_videov2_siglipso400m_2.sh

