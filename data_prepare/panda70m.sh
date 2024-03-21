JOBS_LIMIT=${1:-16}  # Set your limit here

parallel_size=256
idx_size=$(( parallel_size - 1 ))

mkdir -p slurm-logs/data

for idx in $(seq 0 $idx_size); do
    while [ $(jobs -rp | wc -l) -ge $JOBS_LIMIT ]; do
        sleep 1
    done
    echo "Running jobs $(jobs -rp | wc -l) | python llava/data/dataset_impl/panda70m.py --shards=$idx --total=$parallel_size"; 
    srun -A llmservice_nlp_fm \
        -p cpu,cpu_1,cpu_long -t 4:00:00 -J cleanup-$idx-of-$parallel_size \
        --cpus-per-task 2 \
        --mem-per-cpu 64G \
        -e slurm-logs/data/$idx-of-$parallel_size.err \
        -o slurm-logs/data/$idx-of-$parallel_size.txt \
        python llava/data/dataset_impl/panda70m.py --shards=$idx --total=$parallel_size &

done

# --mem-per-cpu