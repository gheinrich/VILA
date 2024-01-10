source ~/.bashrc
conda activate openflamingo
which python

cd ~/workspace/LLaVA/

checkpoint=$1


for i_gpu in {0..7}
do 
    CUDA_VISIBLE_DEVICES=$i_gpu python llava/eval/eval_mmbench.py --model-name $checkpoint --chunk-idx $i_gpu --num-chunks 8 &
done

wait


# counter=0

# for n_shot in 0 2 4 8
# do
#     echo "Evaluting $n_shot-shot model..."
#     CUDA_VISIBLE_DEVICES=$counter python llava/eval/eval_benchmarks.py --model_name $checkpoint --num_samples $n_samples --num_shots $n_shot --conv_version $conv_version &
#     counter=$(( counter + 1 ))
# done

# wait
