#!/bin/bash
#SBATCH --job-name=nvr_lpr_aiagent-vlm:pretraining-siglipso400m-eval-msvd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=interactive,grizzly,polar,grizzly2,polar2
#SBATCH --dependency=singleton
#SBATCH --output=eval-msvd-score.out

CKPT_NAME=vicuna-7b-siglipso400m-ccs-coyo_25m_mmc4core_sharegpt4v_valley-finetune-vflan_sharegpt4v_sft_video_chatgpt_nv_video_flan-e111

srun --label bash /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/v1_5/eval/video_chatgpt/eval_qa_msvd.sh ${CKPT_NAME}



