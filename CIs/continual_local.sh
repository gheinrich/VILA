# /bin/bash
# partition=nvr_elm_llm
partition=llmservice_nlp_fm
report_file=dev/tmp.md

export VILA_CI_RECIPIENTS="ligengz@nvidia.com,jasonlu@nvidia.com,yunhaof@nvidia.com,fuzhaox@nvidia.com"
# a@nvidia.com,b@nvidia.com,c@nvidia.com

> $report_file

# for pyfile in tests/cpu_tests/*.py; do 
for pyfile in $(find ./tests/ -iname "*.py" -not -path "./tests/gpu_tests/*" | xargs); do
    # bash CIs/test_single.sh $pyfile
    pylog=${pyfile//\//\-\-}
    > dev/$pylog.err 
    > dev/$pylog.out
    srun -A $partition \
        -p cpu,cpu_long -t 4:00:00 -J vila-CI:$pyfile \
        -e dev/$pylog.err -o dev/$pylog.out \
        bash CIs/test_single.sh $pyfile &
done

# for pyfile in tests/gpu_tests/*.py; do 
for pyfile in $(find ./tests/gpu_tests -iname "*.py" | xargs); do
    # bash CIs/test_single.sh $pyfile
    pylog=${pyfile//\//\-\-}
    > dev/$pylog.err 
    > dev/$pylog.out
    srun -A $partition \
        -p adlr_services,adlr-debug-batch_block4,batch_block1,batch_block2,batch_block3,batch_block4,batch_singlenode \
        -t 4:00:00 -J vila-CI:$pyfile \
        --gpus-per-node 8 --exclusive \
        -e dev/$pylog.err -o dev/$pylog.out \
        bash CIs/test_single.sh $pyfile &
done

wait

clear
cat $report_file
python CIs/send_email.py --text=$report_file
