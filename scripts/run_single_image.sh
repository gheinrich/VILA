#!/bin/bash
srun --label -A nvr_lpr_aiagent -N 1 -t 4:00:00 -J nvr_lpr_aiagent-vlm:activitynet_inference \
    --gpus-per-node 8 --exclusive \
    --pty bash /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/multi-modality-research/VILA/scripts/inference_single_image.sh
