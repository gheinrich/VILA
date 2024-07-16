slurm_account=${SLURM_ACCOUNT:-nvr_elm_llm}
DATA_PATH=${1:-"/home/ligengz/nvr_elm_llm/dataset/panda70m/webdataset"}

parallel_size=4
idx_size=$(( parallel_size - 1 ))

echo "$slurm_account $DATA_PATH"
for idx in $(seq 0 $idx_size); do
    echo "python llava/data/simple_vila_webdataset.py $DATA_PATH --idx $idx --total $parallel_size"
    srun -A $slurm_account \
        -p cpu,cpu_long -t 4:00:00 -J vila:WIDS-$idx-of-$parallel_size \
        python llava/data/simple_vila_webdataset.py $DATA_PATH --idx $idx --total $parallel_size &
done
wait
python llava/data/simple_vila_webdataset.py $DATA_PATH
