partition=nvr_elm_llm
report_file=dev/tmp.md

export VILA_CI_RECIPIENTS="ligengz@nvidia.com"

> $report_file
for pyfile in tests/cpu_tests/*.py; do 
    # bash CIs/regression_single.sh $pyfile
    srun -A nvr_elm_llm \
        -p cpu -t 4:00:00 -J nvr_elm_llm:dev \
        bash CIs/test_single.sh $pyfile &
done

wait

python CIs/send_email.py --text=dev/tmp.md