#!/bin/bash
#SBATCH --job-name=nextqa-eval:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=1:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=cpu,interactive,grizzly,polar,,polar2,polar3,polar4
#SBATCH --dependency=singleton
#SBATCH --output=eval-nextqa-score.out

CKPT_NAME=vicuna-13b-siglipso400m-ccsvideo-coyo_25m_mmc4core_sharegpt4v_internvid_10M-finetune-baseline_nv_video_flan_jukin_shot2story_shot_only-e11113

srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_nextqa.sh ${CKPT_NAME}



