#!/bin/bash
#SBATCH -A llmservice_nlp_fm            #account
#SBATCH -p interactive,batch_block1,batch_block3,batch_block4,batch_singlenode
#SBATCH -t 04:00:00             		#wall time limit, hr:min:sec
#SBATCH -N 1                    		#number of nodes
#SBATCH -J vila:eval-video-mmev2	#job name
#SBATCH --gpus-per-node 2
#SBATCH --cpus-per-task 12
#SBATCH --mem-per-cpu 16G
#SBATCH --array=0-31
#SBATCH -e slurm-logs/sbatch/dev.err
#SBATCH -o slurm-logs/sbatch/dev.out
#SBATCH --dependency singleton

# cosmos_misc
# llmservice_nlp_fm
# nvr_elm_llm
#### 123 SBATCH --dependency singleton

idx=$SLURM_ARRAY_TASK_ID
total=$SLURM_ARRAY_TASK_COUNT
jname=seval-$idx-of-$total-random

ckpt=${1:-"Efficient-Large-Model/VILA1.5-3b"}
# llava_v1
# hermes-2
model_name=$2
conv_mode=${3:-"hermes-2"}
temperature=${4:-"0.0"}
num_beams=${5:-1}

OUTDIR=slurm-logs/$ckpt
#_$wname
> $OUTDIR/$jname.err
> $OUTDIR/$jname.out


srun \
    -e $OUTDIR/$jname.err -o $OUTDIR/$jname.out \
    python llava/eval/video_mme/video_eval.py \
        --model-path $ckpt --shard $idx --total $total --conv-mode $conv_mode \
        --output_dir=./eval_output/$model_name/video_mme/ --output_name=$model_name --temperature $temperature --num-beams $num_beams