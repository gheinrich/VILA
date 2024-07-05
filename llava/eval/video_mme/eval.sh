# while [ $(squeue --me --name vila:eval-video-mmev2 --noheader | wc -l) -gt 0 ]; do
#     sleep 5
# done

# passed by via environment variables
conv_mode=${conv_mode:-"hermes-2"}

# passed by via positional args
ckpt=${1:-/home/jasonlu/workspace/VILA-Internal/checkpoints/vila-yi-34b-intern-6b-stage2_5_r620_sft_more_r2}
mname=$(echo $ckpt | rev | cut -d "/" -f 1 | rev)

# sbatch -p interactive,interactive_singlenode,$SLURM_PARTITION llava/data_aug/seval.sh Efficient-Large-Model/VILA1.5-3b
sbatch -p interactive,$SLURM_PARTITION -J videomme:$mname \
    llava/eval/video_mme/sbatch_eval.sh \
    $ckpt \
    $mname \
    hermes-2

while [ $(squeue --me --name videomme:$mname --noheader | wc -l) -gt 0 ]; do
    echo "waiting $mname"
    sleep 5
done

ckpt=${1:-/home/jasonlu/workspace/VILA-Internal/checkpoints/vila-yi-34b-intern-6b-stage2_5_r620_sft_more_r2}
mname=$(echo $ckpt | rev | cut -d "/" -f 1 | rev)
# convert the checkpoints
python llava/eval/video_mme/video_eval.py \
    --model-path $ckpt \
    --output_dir eval_output/$mname/video_mme \
    --output_name $mname.json \
    -c

YOUR_RESULTS_FILE=eval_output/$mname/video_mme/${mname}_converted.json
VIDEO_DURATION_TYPE=short,medium,long
python llava/eval/video_mme/mme_calc.py \
    --results_file $YOUR_RESULTS_FILE \
    --video_duration_type $VIDEO_DURATION_TYPE

exit 0
python llava/eval/video_mme/mme_calc.py \
    --results_file $YOUR_RESULTS_FILE \
    --video_duration_type $VIDEO_DURATION_TYPE \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy