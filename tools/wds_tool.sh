#/bin/bash
set -e

idx=${1:-0}
steps=${2:-20}

# rm -rfv /home/ligengz/nvr_elm_llm/dataset/vila-sft-tar/$idx
python tools/create_wds.py --start $idx --output_folder /home/ligengz/nvr_elm_llm/dataset/vila-sft-tar/$idx --seg $steps
rsync -avP /home/ligengz/nvr_elm_llm/dataset/vila-sft-tar/$idx/ login-eos:/home/ligengz/nvr_elm_llm/dataset/vila-sft-tar
rm -rfv /home/ligengz/nvr_elm_llm/dataset/vila-sft-tar/$idx


exit 0

rm -rfv /home/ligengz/nvr_elm_llm/dataset/vila-sft-tar
for idx in $(seq 0 20 520); do
    while [ $(jobs -p | wc -l) -ge 16 ]; do
        sleep 5
    done
    echo "[submit to slurm] Processing $file ";
    srun --account nvr_elm_llm --partition cpu,cpu_long,cpu_short,cpu_interactive,interactive \
        --job-name nvr_elm_llm:wds-$idx \
        --output runs/wds/gen-$idx/%J.out \
        --error runs/wds/gen-$idx/%J.err \
        --time 4:00:00 --exclusive \
        bash tools/wds_tool.sh $idx  &
done
