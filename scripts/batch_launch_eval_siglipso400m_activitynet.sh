#!/bin/bash
#SBATCH --job-name=nvr_lpr_aiagent-vlm:pretraining-siglipso400m-eval-activitynet
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --output=eval_score_activitynet.log

    
srun --label bash ~/workspace/VILA-Internal/scripts/eval/video_chatgpt/eval_qa_activitynet.sh vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-vflan_sharegpt4v_sft_nv_video_flan-linear-e1010

