TOTAL_RUNS=8

for i in $(seq 1 $TOTAL_RUNS); do
    sbatch ~/workspace/VILA-Internal/scripts/batch_launch_train_siglipso400m_3.sh
done