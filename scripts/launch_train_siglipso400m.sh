TOTAL_RUNS=10

for i in $(seq 1 $TOTAL_RUNS); do
srun --label -A nvr_lpr_aiagent -N 32 -t 4:00:00 -J nvr_lpr_aiagent-vlm:pretraining-siglipso400m \
    --gpus-per-node 8 --exclusive  --dependency=singleton --partition=batch_block1 \
    --pty bash /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/vicuna/1_train_mmc4_coyo_sharegpt4v_vicuna_siglipso400m.sh
done
