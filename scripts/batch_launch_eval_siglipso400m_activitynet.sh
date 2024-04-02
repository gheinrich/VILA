#!/bin/bash
#SBATCH --job-name=activitynet-eval:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --dependency=singleton
#SBATCH --partition=grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --output=eval-activitynet-score.log

CKPT_NAME=vicuna-13b-siglipso400m-ccs-coyo_25m_mmc4core_sharegpt4v_valley-finetune-vflan_sharegpt4v_sft_video_chatgpt_nv_video_flan-e4
srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_activitynet.sh ${CKPT_NAME}
