#!/bin/bash
#SBATCH --job-name=nvr_lpr_aiagent-vlm:pretraining-siglipso400m-eval-activitynet
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly
#SBATCH --exclusive
#SBATCH --output=eval_activitynet.log

    
srun --label bash /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/eval/video_chatgpt/run_qa_activitynet.sh vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-vflan_sharegpt4v_sft_nv_video_flan-linear-e1010

