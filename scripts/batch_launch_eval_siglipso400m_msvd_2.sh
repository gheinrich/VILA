#!/bin/bash
#SBATCH --job-name=internvid-pretraining:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=interactive,grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --dependency=singleton
#SBATCH --output=eval-msvd-score-2.out

CKPT_NAME=vicuna-7b-siglipso400m-ccs-coyo_25m_mmc4core_sharegpt4v_valley-finetune-vflan_sharegpt4v_sft_video_chatgpt_nv_video_flan-e111

srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/eval_qa_msvd.sh ${CKPT_NAME}



