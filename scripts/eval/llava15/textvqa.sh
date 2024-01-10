source ~/.bashrc
conda activate vila
which python

if [ ! -d "/tmp/textvqa" ]; then
    echo "Preparing dataset..."
    unzip -q /home/jil/workspace/LLaVA/playground/data/eval/textvqa/train_val_images.zip -d /tmp/textvqa
    echo "done"
else
    echo "Data already exists..."
fi

cd ~/workspace/multi-modality-research/VILA/


python -m llava.eval.eval_llava15_vqa \
    --model_name $1 \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /tmp/textvqa/train_images \
    --answers-file $1/eval/llava15/textvqa.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1_1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file $1/eval/llava15/textvqa.jsonl
