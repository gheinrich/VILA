# 32x40

for i in {64..81}
do
    # echo $i
    start=$((i*45))
    end=$(((i+1)*45))
    echo $start $end
    python llava/data/ccs_to_pkl.py $start $end &
done

wait
