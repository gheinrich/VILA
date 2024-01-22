#!/bin/bash

# Generate Response
# srun --label -A nvr_lpr_aiagent -N 1 -t 4:00:00 -J nvr_lpr_aiagent-vlm:activitynet_inference \
#     --gpus-per-node 8 --exclusive  --partition=batch_block1 \
#     --pty bash /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/eval/video_chatgpt/run_qa_msvd.sh vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-bsz512

# Score the output
srun --label -A nvr_lpr_aiagent -N 1 -t 4:00:00 -J nvr_lpr_aiagent-vlm:activitynet_inference \
    --gpus-per-node 1  --partition=batch_block1 \
    --pty bash /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/eval/video_chatgpt/eval_qa_msvd.sh vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-bsz512

# vicuna-13b-clip336-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-e1010
# vicuna-7b-siglip384-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-e1010
# vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-e1010
# vicuna-7b-siglipso400m-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-bsz512