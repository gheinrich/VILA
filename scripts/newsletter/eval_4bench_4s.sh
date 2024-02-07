
#source ~/.bashrc
#conda activate openflamingo
#which python

# source ~/.bashrc
# source /lustre/fsw/portfolios/llmservice/users/dannyy/lib/anaconda3/etc/profile.d/conda.sh
# conda activate vila-cvpr
which python

#if [ ! -d "/tmp/coco" ]; then
#    echo "Preparing dataset..."
#    tar -xf ~/datasets/coco.tar --directory /tmp/ &
#    unzip -q ~/datasets/textvqa/train_val_images.zip -d /tmp/textvqa &
#    tar -xf ~/datasets/coco.tar --directory /tmp/ &
#    unzip -q ~/datasets/flickr30k/flickr30k-images.zip -d /tmp/flickr &
#    wait
#    echo "done"
#else
#    echo "Data already exists..."
#fi

if [ ! -d "/tmp/coco" ]; then
    echo "Preparing dataset..."
    tar -xf /lustre/fsw/portfolios/llmservice/users/dannyy/vila/datasets-bkp/coco.tar --directory /tmp/ &
    wait
    echo "done"
else
    echo "Data already exists..."
fi

if [ ! -d "/tmp/textvqa" ]; then
    echo "Preparing dataset..."
    unzip -q /lustre/fsw/portfolios/llmservice/users/dannyy/vila/datasets-bkp/textvqa/train_val_images.zip -d /tmp/textvqa &
    wait
    echo "done"
else
    echo "Data already exists..."
fi

if [ ! -d "/tmp/fickr" ]; then
    echo "Preparing dataset..."
    unzip -q /lustre/fsw/portfolios/llmservice/users/dannyy/vila/datasets-bkp/flickr30k/flickr30k-images.zip -d /tmp/flickr &
    wait
    echo "done"
else
    echo "Data already exists..."
fi

#cd ~/workspace/LLaVA/
# cd /lustre/fsw/portfolios/llmservice/users/dannyy/vila/vila-dev/github_version/VILA

checkpoint=$1
conv_version=${2:-"vicuna_v1_1_nosys"}
n_samples=${3:-5000}
num_beams=${4:-1}

CUDA_VISIBLE_DEVICES=0 python llava/eval/eval_benchmarks.py --dataset_name okvqa --model_name $checkpoint --num_samples $n_samples --num_shots 4 --conv_version $conv_version --num_beams $num_beams &
CUDA_VISIBLE_DEVICES=1 python llava/eval/eval_benchmarks.py --dataset_name textvqa --model_name $checkpoint --num_samples $n_samples --num_shots 4 --conv_version $conv_version &
# CUDA_VISIBLE_DEVICES=2 python llava/eval/eval_benchmarks.py --model_name $checkpoint --num_samples $n_samples --num_shots 4 --conv_version $conv_version &
CUDA_VISIBLE_DEVICES=3 python llava/eval/eval_benchmarks.py --dataset_name flickr --model_name $checkpoint --num_samples $n_samples --num_shots 4  --conv_version $conv_version&

wait
