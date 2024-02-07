#!/bin/bash
#SBATCH --job-name=nvr_lpr_aiagent-vlm:pretraining-siglipso400m-eval-activitynet
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly
#SBATCH --output=eval_score_activitynet.log

    
srun --label bash /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/eval/video_chatgpt/eval_qa_activitynet.sh vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-vflan_sharegpt4v_sft_nv_video_flan-linear-e1010

