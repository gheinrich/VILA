#!/bin/bash
#SBATCH --job-name=nvr_lpr_aiagent-vlm:pretraining-siglipso400m-eval-msvd
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=interactive,grizzly,polar,grizzly2,polar2
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=eval-msvd-inference.out
    
CKPT_NAME=vicuna-7b-siglipso400m-ccs-coyo_25m_mmc4core_sharegpt4v_valley-finetune-vflan_sharegpt4v_sft_video_chatgpt_nv_video_flan-e111
model_path=/lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/ckpts/${CKPT_NAME}
srun --label bash /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/v1_5/eval/video_chatgpt/run_qa_msvd.sh ${model_path} ${CKPT_NAME}


