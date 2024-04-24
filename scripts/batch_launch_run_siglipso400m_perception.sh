#!/bin/bash
#SBATCH --job-name=perception-eval:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=interactive,grizzly,polar,,polar2,polar3,polar4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=eval-perception-inference.out
    
# CKPT_NAME=vicuna-13b-siglipso400m-ccs-coyo_25m_mmc4core_sharegpt4v_valley-finetune-vflan_sharegpt4v_sft_video_chatgpt_nv_video_flan-e4
CKPT_NAME=vicuna-13b-siglipso400m-ccsvideo-coyo_25m_mmc4core_sharegpt4v_internvid_10M-finetune-baseline_nv_video_flan_jukin_shot2story_shot_only-e11113
model_path=./checkpoints/${CKPT_NAME} 

srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/run_qa_perception.sh ${model_path} ${CKPT_NAME}


