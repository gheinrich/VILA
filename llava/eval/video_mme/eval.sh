#/bin/bash
set -e

# passed by via environment variables
conv_mode=${conv_mode:-"hermes-2"}
# passed by via positional args
ckpt=${1:-Efficient-Large-Model/VILA1.5-40b}
mname=$(echo $ckpt | rev | cut -d "/" -f 1 | rev)

export temperature=0
export num_beams=1
export num_video_frames=12
export conv_mode=${conv_mode:-"hermes-2"}

jname=videomme:$mname-f$num_video_frames
echo "launching $jname"

# hook function to handle ctrl+c interrupt and cancel jobs
exitfn () {
    trap SIGINT  # Restore signal handling for SIGINT.
    echo; echo 'Aarghh!!'
    echo 'Canceling jobs' $jname
    scancel -n $jname
    exit
}
trap "exitfn" INT

sbatch -A nvr_elm_llm -p $SLURM_PARTITION --wait -J $jname \
    llava/eval/video_mme/sbatch_eval.sh \
    $ckpt

ORIG_RESULTS_FILE="runs/eval/$mname/video_mme/frames-${num_video_frames}.json"
YOUR_RESULTS_FILE="runs/eval/$mname/video_mme/frames-${num_video_frames}_converted.json"
VIDEO_DURATION_TYPE="short,medium,long"
python llava/eval/video_mme/convert.py \
    --answer_file $ORIG_RESULTS_FILE \
    --output_file $YOUR_RESULTS_FILE

export WANDB_NAME="$mname-frames-${num_video_frames}"
python llava/eval/video_mme/mme_calc.py \
    --results_file $YOUR_RESULTS_FILE \
    --video_duration_type $VIDEO_DURATION_TYPE \
    --your_answer_key response_w/_sub

python llava/eval/video_mme/mme_calc.py \
    --results_file $YOUR_RESULTS_FILE \
    --video_duration_type $VIDEO_DURATION_TYPE \
    --your_answer_key response_w/o_sub
