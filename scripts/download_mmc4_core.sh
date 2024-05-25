source ~/.bashrc
conda activate llava
which python

cd ~/workspace/LLaVA/llava/data

n_node=$SLURM_JOB_NUM_NODES
echo "number of nodes:" $n_node
echo "node rank:" $SLURM_PROCID

per_process=181
for i in {0..8}
do
    rank=$(( i + $SLURM_PROCID * 8 ))
    start=$(( per_process * rank ))
    end=$(( start + per_process ))
    echo "start:" $start
    echo "end:" $end

    python download_mmc4_core.py $start $end &
done

wait
