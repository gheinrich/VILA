#!/bin/bash
#SBATCH --job-name=activitynet-eval:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --output=eval-activitynet-score.log

CKPT_NAME=vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-bsz512
srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_activitynet.sh ${CKPT_NAME}
