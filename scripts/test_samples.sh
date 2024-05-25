
# caption
python llava/eval/run_vila.py --model-name ~/models/llava/llava-13b-v0/ --image-file /tmp/coco/val2014/COCO_val2014_000000262148.jpg --query "Give a short and clear explanation of the subsequent image."

python llava/eval/run_vila.py --model-name ~/workspace/LLaVA/checkpoints/llava-13b-finetune-cc3m-e1 --image-file /tmp/coco/val2014/COCO_val2014_000000262148.jpg --query "Give a short and clear explanation of the subsequent image." --conv-mode vicuna_v1_1

python llava/eval/run_vila.py --model-name ~/workspace/LLaVA/checkpoints/llava-13b-cc3m-finetune-sharegpt-e1 --image-file /tmp/coco/val2014/COCO_val2014_000000262148.jpg --query "Give a short and clear explanation of the subsequent image." --conv-mode vicuna_v1_1


# vqa
python llava/eval/run_vila.py --model-name ~/models/llava/llava-13b-v0/ --image-file /tmp/coco/val2014/COCO_val2014_000000262148.jpg --query "What is he on top of?"

python llava/eval/run_vila.py --model-name ~/workspace/LLaVA/checkpoints/llava-13b-finetune-cc3m-e1 --image-file /tmp/coco/val2014/COCO_val2014_000000262148.jpg --query "Give a short and clear explanation of the subsequent image."

python llava/eval/run_vila.py --model-name ~/workspace/LLaVA/checkpoints/llava-13b-cc3m-finetune-sharegpt-e1 --image-file /tmp/coco/val2014/COCO_val2014_000000262148.jpg --query "Give a short and clear explanation of the subsequent image."



# 393225

python llava/eval/run_vila.py --model-name ~/models/llava/llava-13b-v0/ --image-file /tmp/coco/val2014/COCO_val2014_000000393225.jpg --query "What is to the right of the soup?"
