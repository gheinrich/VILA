JOBS_LIMIT=2  # Set your limit here
model_id=NousResearch/Llama-2-13b-chat-hf
model_id=NousResearch/Llama-2-70b-chat-hf
for f in captioner/*.json; do
  while [ $(jobs -rp | wc -l) -ge $JOBS_LIMIT ]; do
    sleep 1
  done

  fname=$(echo $f | cut -d "/" -f 2)
  model=$(echo $model_id | cut -d "/" -f 2)

  # Replace this with your actual command
  echo "Processing task $f and running jobs $(jobs -rp | wc -l)"; \
  srun --label -A llmservice_nlp_fm -N 1 \
    -p batch_block1,batch_block2,batch_block3 -t 4:00:00 \
    -J llmservice_nlp_fm-dev:cap2qa-$fname-$model --gpus-per-node 8 --exclusive \
    -e slurm-logs/dev/$fname-$model-$j.err -o slurm-logs/dev/$fname-$model-$j.out \
    torchrun --nproc-per-node 8 llava/data_aug/caption2qa.py --data_path=$f --model_id=$model_id &
done
wait