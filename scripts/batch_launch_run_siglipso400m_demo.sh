#!/bin/bash
#SBATCH --job-name=demo-eval:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=1:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=interactive,grizzly,polar,grizzly2,polar2,polar3,polar4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=eval-demo-inference.out
    
CKPT_NAME=vicuna-13b-siglipso400m-ccs-coyo_25m_mmc4core_sharegpt4v_valley-finetune-vflan_sharegpt4v_sft_video_chatgpt_nv_video_flan-e4
model_path=./checkpoints/${CKPT_NAME} 
PROMPT_TEMPLATE=bin_20_80

srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/run_qa_demo.sh ${model_path} ${CKPT_NAME} ${PROMPT_TEMPLATE}


