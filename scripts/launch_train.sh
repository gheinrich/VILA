TOTAL_RUNS=5

for i in $(seq 1 $TOTAL_RUNS); do
srun --label -A nvr_lpr_aiagent -N 16 -t 4:00:00 -J nvr_lpr_aiagent-vlm:pretraining \
    --gpus-per-node 8 --exclusive  --dependency=singleton --partition=batch_block1 \
    --pty bash ~/workspace/VILA-Internal/scripts/vicuna/2_sft_video.sh
done
