# while [ $(squeue --me --name vila:eval-video-mmev2 --noheader | wc -l) -gt 0 ]; do
#     sleep 5
# done

# sbatch -p interactive,interactive_singlenode,$SLURM_PARTITION llava/data_aug/seval.sh Efficient-Large-Model/VILA1.5-3b
sbatch -p interactive,interactive_singlenode,$SLURM_PARTITION \
    llava/eval/video_mme/sbatch_eval.sh Efficient-Large-Model/VILA1.5-40b hermes-2

while [ $(squeue --me --name vila:eval-video-mmev2 --noheader | wc -l) -gt 0 ]; do
    sleep 5
done
# convert the checkpoints
python llava/data_aug/video_eval.py --model-path Efficient-Large-Model/VILA1.5-40b --convert

YOUR_RESULTS_FILE=VILA1.5-3bmme_bench_dev_converted.json
VIDEO_DURATION_TYPE=short,medium,long
python llava/eval/video_mme/mme_calc.py \
    --results_file $YOUR_RESULTS_FILE \
    --video_duration_type $VIDEO_DURATION_TYPE \
    --return_categories_accuracy \
    --return_sub_categories_accuracy \
    --return_task_types_accuracy