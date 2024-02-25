TOTAL_RUNS=4

for i in $(seq 1 $TOTAL_RUNS); do
    sbatch /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/batch_launch_train_siglipso400m.sh
done
