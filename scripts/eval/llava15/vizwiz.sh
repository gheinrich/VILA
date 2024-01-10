
source ~/.bashrc
conda activate vila
which python

if [ ! -d "/tmp/vizwiz" ]; then
    echo "Preparing dataset..."
    unzip -q /home/jil/workspace/LLaVA/playground/data/eval/vizwiz/test.zip -d /tmp/vizwiz
    echo "done"
else
    echo "Data already exists..."
fi

cd ~/workspace/multi-modality-research/VILA/


python -m llava.eval.eval_llava15_vqa \
    --model_name $1 \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder  /tmp/vizwiz/test \
    --answers-file $1/eval/llava15/vizwiz.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1_1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file $1/eval/llava15/vizwiz.jsonl \
    --result-upload-file $1/eval/llava15/vizwiz-upload.jsonl

