#!/bin/bash
#SBATCH --job-name=demo-eval:nvr_lpr_aiagent
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=1:00:00
#SBATCH -A nvr_lpr_aiagent
#SBATCH --partition=interactive,grizzly,polar,polar2,polar3,polar4
#SBATCH --exclusive
#SBATCH --dependency=singleton
#SBATCH --output=eval-demo-inference.out

CKPT_NAME=vilavideo-sft-7b-v032-fixedfps
# vilavideo-sft-7b-v032
model_path=./checkpoints/${CKPT_NAME}
PROMPT_TEMPLATE=bin_20_80

srun --label bash ~/workspace/VILA-Internal/scripts/v1_5/eval/video_chatgpt/run_qa_demo.sh ${model_path} ${CKPT_NAME} ${PROMPT_TEMPLATE}
