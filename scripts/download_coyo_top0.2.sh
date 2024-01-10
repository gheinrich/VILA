source ~/.bashrc
conda activate llava
which python

cd ~/workspace/LLaVA/llava/data

n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $SLURM_PROCID

for i in {0..7}
do
    rank=$(( i + $SLURM_PROCID * 8 ))
    echo "rank:" $rank

    python process_coyo.py $rank first &
done

wait
