#!/bin/bash
#SBATCH --job-name=activitynet-eval:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --output=eval-activitynet-score-2.log

CKPT_NAME=vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-vflan_sharegpt4v_sft_nv_video_flan-linear-e1010
srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_activitynet.sh ${CKPT_NAME}
