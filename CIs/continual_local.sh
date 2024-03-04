partition=nvr_elm_llm
report_file=dev/tmp.md

export VILA_CI_RECIPIENTS="ligengz@nvidia.com"
# a@nvidia.com,b@nvidia.com,c@nvidia.com

> $report_file
for pyfile in tests/cpu_tests/*.py; do 
    # bash CIs/test_single.sh $pyfile
    srun -A nvr_elm_llm \
        -p cpu -t 4:00:00 -J vila-CI:$pyfile \
        bash CIs/test_single.sh $pyfile &
done


for pyfile in tests/gpu_tests/*.py; do 
    # bash CIs/test_single.sh $pyfile
    srun -A nvr_elm_llm \
        -p batch_block1,batch_block2,batch_block3,batch_block4,interactive -t 4:00:00 -J vila-CI:$pyfile \
        --gpus-per-node 8 --exclusive \
        bash CIs/test_single.sh $pyfile &
done

wait

clear
cat $report_file
# python CIs/send_email.py --text=dev/tmp.md