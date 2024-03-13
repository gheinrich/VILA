partition=${SLURM_PARTITION:-nvr_elm_llm}
data_path=${1:"~/nvr_elm_llm/dataset/coyo-25m-vila"}

for idx in $(seq 0 9); do 
    srun -A $partition \
        -p cpu,cpu_long -t 4:00:00 -J vila:WIDS-$idx \
        python llava/data/simple_vila_webdataset.py $data_path --idx $idx --total 10 &
done
wait
python llava/data/simple_vila_webdataset.py $data_path