source ~/.bashrc
conda activate llava
which python

cd ~/workspace/LLaVA/llava/data/coyo

n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $1

for i in {0..7}
do
    rank=$(( i + $1 * 8 ))
    echo "rank:" $rank

    python process_coyo.py $rank second &
done

wait
