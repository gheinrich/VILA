SPLIT="mmbench_dev_cn_20231003"

source ~/.bashrc
conda activate vila
which python

cd ~/workspace/multi-modality-research/VILA/

python -m llava.eval.model_vqa_mmbench \
    --model-name $1 \
    --question-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --answers-file  $1/eval/llava15/mmbench_cn/$SPLIT.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1_1

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench_cn/$SPLIT.tsv \
    --result-dir $1/eval/llava15/mmbench_cn \
    --upload-dir $1/eval/llava15/mmbench_cn-upload \
    --experiment $SPLIT
