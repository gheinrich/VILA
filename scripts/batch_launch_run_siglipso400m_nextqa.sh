#!/bin/bash
#SBATCH --job-name=nextqa-eval:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=interactive,grizzly,polar,polar2,polar3,polar4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=eval-nextqa-inference.out
    
CKPT_NAME=vila-video-13b-fix
model_path=./checkpoints/${CKPT_NAME} 

srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/run_qa_nextqa.sh ${model_path} ${CKPT_NAME}


