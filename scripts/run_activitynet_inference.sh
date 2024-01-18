#!/bin/bash

## Generate Response
# srun --label -A nvr_lpr_aiagent -N 1 -t 4:00:00 -J nvr_lpr_aiagent-vlm:activitynet_inference \
#     --gpus-per-node 8 --exclusive \
#     --pty bash /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/multi-modality-research/VILA/scripts/eval/video_chatgpt/run_qa_activitynet.sh vicuna-7b-clip336-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-e1004

## Score the output
srun --label -A nvr_lpr_aiagent -N 1 -t 4:00:00 -J nvr_lpr_aiagent-vlm:activitynet_inference \
    --gpus-per-node 1 \
    --pty bash /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/multi-modality-research/VILA/scripts/eval/video_chatgpt/eval_qa_activitynet.sh vicuna-7b-clip336-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-e1004
