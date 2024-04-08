#!/bin/bash
#SBATCH --job-name=perception-eval:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=1:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=cpu,interactive,grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --dependency=singleton
#SBATCH --output=eval-perception-score.out

CKPT_NAME=vicuna-7b-siglipso400m-ccsvideo-coyo_25m_mmc4core_sharegpt4v_internvid_10M-finetune-baseline_nv_video_flan_jukin_shot2story_shot_only-e2

srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_perception.sh ${CKPT_NAME}


