source ~/.bashrc
conda activate openflamingo
which python


if [ ! -d "/tmp/coco" ]; then
    echo "Preparing dataset..."
    tar -xf ~/datasets/coco.tar --directory /tmp/
    echo "done"
else
    echo "Data already exists..."
fi

cd ~/workspace/LLaVA/

checkpoint=$1
conv_version=${2:-"v1"}
n_samples=${3:-5000}
num_beams=${4:-1}

counter=0

for n_shot in 0 2 4 8
do
    echo "Evaluting $n_shot-shot model..."
    CUDA_VISIBLE_DEVICES=$counter python llava/eval/eval_benchmarks.py --dataset_name okvqa --model_name $checkpoint --num_samples $n_samples --num_shots $n_shot --conv_version $conv_version --num_beams $num_beams --verbose --nll_rank &
    counter=$(( counter + 1 ))
done

wait
