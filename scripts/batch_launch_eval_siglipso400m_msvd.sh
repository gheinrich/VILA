#!/bin/bash
#SBATCH --job-name=nvr_lpr_aiagent-vlm:pretraining-siglipso400m-eval-msvd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly
#SBATCH --dependency=singleton
#SBATCH --output=eval-msvd-score.out

    
srun --label bash /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/eval/video_chatgpt/eval_qa_msvd.sh vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-nv_video_flan-linear-e1010



