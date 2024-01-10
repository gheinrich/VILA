total_nums=4

for dataset in sam coco; do
for idx in $(seq 0 $total_nums); do
    srun --label -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm-vlm:raw-$dataset-$idx-$total_nums \
        --gpus-per-node 1 \
        --pty bash ~/workspace/multi-modality-research/VILA/scripts/caption.sh \
        checkpoints/vila-7B \
        $dataset \
        $idx \
        $total_nums &

    srun --label -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm-vlm:cap-$dataset-$idx-$total_nums \
        --gpus-per-node 1 \
        --pty bash ~/workspace/multi-modality-research/VILA/scripts/caption.sh \
        checkpoints/vicuna-13b-clip336-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v-nosqa-linear-captioner1 \
        $dataset \
        $idx \
        $total_nums &

done
wait
done
# rsync -chazvPL draco-oci-login-01.draco-oci-iad.nvidia.com:~/datasets /lustre/fs2/portfolios/nvr/users/ligengz