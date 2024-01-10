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
conv_version=${2:-"vicuna_v1_1_nosys"}
n_samples=${3:-5000}

counter=0

for n_shot in 0 2 4 8
do
    echo "Evaluting $n_shot-shot model..."
    CUDA_VISIBLE_DEVICES=$counter python llava/eval/eval_benchmarks.py --model_name $checkpoint --num_samples $n_samples --num_shots $n_shot --conv_version $conv_version &
    counter=$(( counter + 1 ))
done

wait
