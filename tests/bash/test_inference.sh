#/bin/bash

set -e

vila-infer \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --conv-mode vicuna_v1 \
    --text "Please describe the image" \
    --media inference_test/test_data/caption_meat.jpeg

vila-infer \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --conv-mode vicuna_v1 \
    --text "Please describe the video" \
    --media https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4
