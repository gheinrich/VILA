#!/bin/bash
#SBATCH --job-name=nvr_lpr_aiagent-vlm:pretraining-siglipso400m-eval-msvd
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=grizzly
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=eval-msvd-inference.out
    
srun --label bash /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/eval/video_chatgpt/run_qa_msvd.sh vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-nv_video_flan-linear-e1010


