

python llava/eval/run_llava.py --model-name checkpoints/llama2-7b-mmc4sub-dualflan-finetune-llava+sharegpt-linearclip-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image> Please describe the traffic condition." \
    --image-file "images/av/av0.png" 

python llava/eval/run_llava.py --model-name checkpoints/llama2-7b-mmc4sub-dualflan-finetune-llava+sharegpt-linearclip-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image> What color is the traffic light?" \
    --image-file "images/av/av0.png" 


python llava/eval/run_llava.py --model-name checkpoints/llama2-7b-mmc4sub-dualflan-finetune-llava+sharegpt-linearclip-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image> If you are driving, should you start and proceed through the intersection? Why?" \
    --image-file "images/av/av0.png" 


python llava/eval/run_llava.py --model-name checkpoints/llama2-7b-mmc4sub-dualflan-finetune-llava+sharegpt-linearclip-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image> If you are driving, can you honk at the pedestrians crossing the road? Why?" \
    --image-file "images/av/av1.png" 



python llava/eval/run_llava.py --model-name checkpoints/llama2-7b-mmc4sub-finetune-dualflan-linearclip-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image> Can you briefly explain the content in the image?" \
    --image-file "images/av/av1.png" 

python llava/eval/run_llava.py --model-name checkpoints/llama2-7b-mmc4sub-dualflan-finetune-llava+sharegpt-linearclip-e1-nose --conv-mode vicuna_v1_1 \
    --query "<image> Can the vehicle proceed through the traffic now?" \
    --image-file "images/av/av0.png" 