#!/bin/bash
#SBATCH --job-name=vila-7b-internvid10m-pretraining:nvr_lpr_aiagent
#SBATCH --nodes=32
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,polar2,polar3,polar4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=7b-internvid-10m-training-4-fixedfps.out


# srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/2_train_pretraining_vilavideo7b_v03.sh
srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/video/3_sft_vilavideo7b_v03_fixedfps.sh
