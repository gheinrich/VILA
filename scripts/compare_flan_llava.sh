python llava/eval/run_vila.py --model-name checkpoints/vicuna-7b-finetune-vflan+tflan-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "How many legs do the animals in the picture have in total? Think step by step."     --image-file "images/puppy_and_kitten.webp"
# The dogs have 4 legs each and the cats have 4 legs each. So, the total number of legs the animals have is 4 + 4 = 8. So, the answer is 8.


python llava/eval/run_vila.py --model-name checkpoints/vicuna-7b-finetune-vflan+tflan-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "Write a poem for the scene."     --image-file "images/puppy_and_kitten.webp"
# A dog and a cat are playing together outside. The dog is black and white and the cat is brown and white. They are playing with a toy in the grass.

python llava/eval/run_vila.py --model-name checkpoints/vicuna-7b-finetune-vflan+tflan-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "Write a haiku for the scene."     --image-file "images/puppy_and_kitten.webp"
# A dog and a cat are playing in the grass.



python llava/eval/run_vila.py --model-name checkpoints/vicuna-7b-dualflan-finetune-llava+lrv+sharegpt-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "How many legs do the animals in the picture have in total? Think step by step."     --image-file "images/puppy_and_kitten.webp"
# The dog has four legs, and the cat has four legs. So, the total number of legs is 8.

python llava/eval/run_vila.py --model-name checkpoints/vicuna-7b-dualflan-finetune-llava+lrv+sharegpt-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "Write a poem for the scene."     --image-file "images/puppy_and_kitten.webp"


python llava/eval/run_vila.py --model-name checkpoints/vicuna-7b-dualflan-finetune-llava+sharegpt-downsampleprojse-e1-nose --conv-mode vicuna_v1_1 \
    --query "Write a poem for the scene."     --image-file "images/puppy_and_kitten.webp"
# In a field of green, two young animals play,
# A dog and a kitten, both so fair.
# The dog's eyes are bright, and its tail wags,
# While the kitten's fur is soft and tucked.

# They sniff and touch, and laugh and play,
# The dog's wagging tail, the kitten's way.
# Their laughter echoes, a joyful sound,
# As they share this moment, both in the ground.

# The sun shines bright, and the sky is clear,
# A perfect day for these two to be here.
# They may be different, but they share a bond,
# A friendship that grows, as they play and bond.

# So let us cherish this moment, in time,
# A scene of love and joy, a sight divine.