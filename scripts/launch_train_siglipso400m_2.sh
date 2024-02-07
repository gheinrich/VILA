TOTAL_RUNS=8

for i in $(seq 1 $TOTAL_RUNS); do
    sbatch /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/batch_launch_train_siglipso400m_2.sh
done
