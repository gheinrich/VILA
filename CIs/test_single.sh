pyfile=${1:-tests/cpu_tests/success.py}
report_file=${2:-dev/tmp.md}

python $pyfile; 
if [ $? -eq 0 ]; then
    msg="[CIs] $pyfile succeeded"
else
    msg="[CIs] $pyfile failed"
fi

echo "$msg"
echo "$msg" >> $report_file
