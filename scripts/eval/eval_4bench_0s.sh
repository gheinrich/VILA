source ~/.bashrc
conda activate openflamingo
which python


if [ ! -d "/tmp/coco" ]; then
    echo "Preparing dataset..."
    tar -xf ~/datasets/coco.tar --directory /tmp/ &
    unzip -q ~/datasets/textvqa/train_val_images.zip -d /tmp/textvqa &
    tar -xf ~/datasets/coco.tar --directory /tmp/ &
    unzip -q ~/datasets/flickr30k/flickr30k-images.zip -d /tmp/flickr &
    wait
    echo "done"
else
    echo "Data already exists..."
fi

cd ~/workspace/LLaVA/

checkpoint=$1
conv_version=${2:-"vicuna_v1_1_nosys"}
n_samples=${3:-5000}
num_beams=${4:-1}

CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_benchmarks.py --dataset_name okvqa --model_name $checkpoint --num_samples $n_samples --num_shots 0 --conv_version $conv_version --num_beams $num_beams &
CUDA_VISIBLE_DEVICES=1 python llava/eval/eval_benchmarks.py --dataset_name textvqa --model_name $checkpoint --num_samples $n_samples --num_shots 0 --conv_version $conv_version &
CUDA_VISIBLE_DEVICES=2 python llava/eval/eval_benchmarks.py --model_name $checkpoint --num_samples $n_samples --num_shots 0 --conv_version $conv_version &
CUDA_VISIBLE_DEVICES=3 python llava/eval/eval_benchmarks.py --dataset_name flickr --model_name $checkpoint --num_samples $n_samples --num_shots 0  --conv_version $conv_version&

wait
