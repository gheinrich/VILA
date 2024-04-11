# /bin/bash
# partition=nvr_elm_llm
# partition=llmservice_nlp_fm
SLURM_ACCOUNT=${SLURM_ACCOUNT:-"llmservice_nlp_fm"}
SLURM_PARTITION=${SLURM_PARTITION:-"adlr-debug-batch_block4,batch_block1,batch_block2,batch_block3,batch_block4"}

report_file=dev/tmp.md
mkdir -p dev

_VILA_CI_RECIPIENTS=${1:-""}
# a@nvidia.com,b@nvidia.com,c@nvidia.com

if [ -n "$_VILA_CI_RECIPIENTS" ]; then
    echo "Sending reports to $_VILA_CI_RECIPIENTS"
else
    echo "Printing reports locally."
fi

# exit 0
> $report_file

# NOTE(ligeng): jukinmedia requires large memory to load.
# --cpus-per-task 8 \
# --mem-per-cpu 16G \
for pyfile in $(find ./tests/ -iname "*.py" -not -path "./tests/gpu_tests/*" -not -path "./tests/archive_tests/*" | xargs); do
    echo "[cpu] testing $pyfile"
    pylog=${pyfile//\//\-\-}
    # clear two files
    > dev/$pylog.err 
    > dev/$pylog.out
    srun -A $SLURM_ACCOUNT \
        -p cpu,cpu_long -t 4:00:00 -J vila-CI:$pyfile \
        --exclusive \
        -e dev/$pylog.err -o dev/$pylog.out \
        bash CIs/test_single.sh $pyfile $report_file &
done


# for pyfile in $(find ./tests/gpu_tests -iname "*.py" | xargs); do
#     echo "[gpu] testing $pyfile"
#     pylog=${pyfile//\//\-\-}
#     > dev/$pylog.err 
#     > dev/$pylog.out
#     srun -A $SLURM_ACCOUNT \
#         -p $SLURM_PARTITION,batch_singlenode \
#         -t 4:00:00 -J vila-CI:$pyfile \
#         --gpus-per-node 8 --exclusive \
#         -e dev/$pylog.err -o dev/$pylog.out \
#         bash CIs/test_single.sh $pyfile $report_file &
# done

# for pyfile in $(find ./tests/bash_tests -iname "*.sh" | xargs); do
#     echo "[bash] testing $pyfile"
#     pylog=${pyfile//\//\-\-}
#     > dev/$pylog.err 
#     > dev/$pylog.out
#     srun -A $SLURM_ACCOUNT \
#         -p $SLURM_PARTITION,batch_singlenode \
#         -t 4:00:00 -J vila-CI:$pyfile \
#         --gpus-per-node 8 --exclusive \
#         -e dev/$pylog.err -o dev/$pylog.out \
#         bash CIs/test_single.sh $pyfile $report_file bash &
# done

wait
clear

if [ -n "$_VILA_CI_RECIPIENTS" ]; then
    python CIs/send_email.py --text=$report_file --recipients $_VILA_CI_RECIPIENTS
else
    cat $report_file
fi
