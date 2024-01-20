MAX_JOBS=4
num_nodes=20
echo "Running on $num_nodes"

end_idx=$(($num_nodes-1))
# srun --label -A nvr_elm_llm -N $num_nodes -t 4:00:00 -J nvr_elm_llm-vlm:label-coyo \
#     --gpus-per-node 8 --exclusive \
#     --dependency singleton \
#     --pty bash ~/workspace/multi-modality-research/VILA/scripts/caption.sh

for i in $(seq 0 $end_idx); do 
# for i in {1..10}; do
    srun --label -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm-vlm:label-coyo-$i-$num_nodes \
        --gpus-per-node 1 \
        --pty bash ~/workspace/multi-modality-research/VILA/scripts/caption.sh $i $num_nodes &
    # echo "Testing $i" && sleep $i &

    while [ $(jobs -rp | wc -l) -ge $MAX_JOBS ]; do
        sleep 5
    done

done


# total_nums=4
# for dataset in sam coco; do
# for idx in $(seq 0 $total_nums); do
#     srun --label -A nvr_elm_llm -N 1 -t 4:00:00 -J nvr_elm_llm-vlm:raw-$dataset-$idx-$total_nums \
#         --gpus-per-node 1 \
#         --pty bash ~/workspace/multi-modality-research/VILA/scripts/caption.sh \
#         /home/jasonlu/workspace/multi-modality-research/VILA/checkpoints/vicuna-13b-clip336-mmc4sub+coyo-finetune-captioner-e4 \
#         $dataset \
#         $idx \
#         $total_nums &
# done
# wait
# done
# checkpoints/vila-7B
# /home/jasonlu/workspace/multi-modality-research/VILA/checkpoints/vicuna-13b-clip336-mmc4sub+coyo-finetune-captioner-e4
# rsync -chazvPL draco-oci-login-01.draco-oci-iad.nvidia.com:~/datasets /lustre/fs2/portfolios/nvr/users/ligengz