pyfile=${1:-tests/cpu_tests/success.py}
report_file=${2:-dev/tmp.md}

python $pyfile; 
if [ $? -eq 0 ]; then
    msg="[succeeded] $pyfile "
else
    msg="[failed] $pyfile "
fi

echo "$msg"
echo "$msg" >> $report_file
