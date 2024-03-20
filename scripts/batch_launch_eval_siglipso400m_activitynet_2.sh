#!/bin/bash
#SBATCH --job-name=activitynet-eval-2:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --dependency=singleton
#SBATCH --partition=grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --output=eval-activitynet-score-2.log

CKPT_NAME=vicuna-7b-siglipso400m-ccs-coyo_25m_mmc4core_sharegpt4v_internvid_10M-finetune-baseline_nv_video_flan_jukin_shot2story_shot_only-e1
srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_activitynet.sh ${CKPT_NAME}
