# conda activate openflamingo

num_beams=${2:-1}

python llava/eval/model_vqa.py \
    --model-name checkpoints/$1 \
    --question-file /home/jil/datasets/llava-bench-in-the-wild/questions.jsonl \
    --image-folder  /home/jil/datasets/llava-bench-in-the-wild/images \
    --answers-file checkpoints/$1/bench-beam${num_beams}.jsonl \
    --num_beams $num_beams
