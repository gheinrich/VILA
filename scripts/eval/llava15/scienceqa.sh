source ~/.bashrc
conda activate vila
which python

if [ ! -d "/tmp/scienceqa" ]; then
    echo "Preparing dataset..."
    unzip -q /home/jil/workspace/LLaVA/playground/data/eval/scienceqa/test.zip -d /tmp/scienceqa
    echo "done"
else
    echo "Data already exists..."
fi

cd ~/workspace/VILA/


python -m llava.eval.model_vqa_science \
    --model-name $1 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /tmp/scienceqa/test \
    --answers-file $1/eval/llava15/scienceqa.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1_1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file $1/eval/llava15/scienceqa.jsonl \
    --output-file $1/eval/llava15/scienceqa_output.jsonl \
    --output-result $1/eval/llava15/scienceqa_result.jsonl

    
