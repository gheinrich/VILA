TOTAL_RUNS=1

for i in $(seq 1 $TOTAL_RUNS); do
srun --label -A nvr_lpr_aiagent -N 16 -t 4:00:00 -J nvr_lpr_aiagent-vlm:pretraining-siglipso400m \
    --gpus-per-node 8 --exclusive  --dependency=singleton --partition=batch_block1 \
    --pty bash /lustre/fs2/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/VILA/scripts/vicuna/2_sft_video_siglipso400m.sh
done
