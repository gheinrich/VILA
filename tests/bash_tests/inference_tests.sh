# TODO(ligeng): replace with the model with finalized 7B model.
python llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/CI-new-format-llama7b-siglip \
    --query "Please describe the image" \
    --image-file inference_test/test_data/caption_meat.jpeg

python llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/CI-new-format-llama7b-siglip \
    --query "Please describe the video" \
    --video-file https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4