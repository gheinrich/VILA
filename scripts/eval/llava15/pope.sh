
source ~/.bashrc
conda activate vila
which python


if [ ! -d "/tmp/coco" ]; then
    echo "Preparing dataset..."
    tar -xf ~/datasets/coco.tar --directory /tmp/
    echo "done"
else
    echo "Data already exists..."
fi

cd ~/workspace/multi-modality-research/VILA/

python -m llava.eval.eval_llava15_vqa \
    --model_name $1 \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /tmp/coco/val2014 \
    --answers-file  $1/eval/llava15/pope.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1_1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $1/eval/llava15/pope.jsonl
