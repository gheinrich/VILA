SPLIT="mmbench_dev_20230712"

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/VILA/

python -m llava.eval.model_vqa_mmbench \
    --model-name $1 \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file  $1/eval/llava15/mmbench/$SPLIT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1_1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir $1/eval/llava15/mmbench \
    --upload-dir $1/eval/llava15/mmbench-upload \
    --experiment $SPLIT
