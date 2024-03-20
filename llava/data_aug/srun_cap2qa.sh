# huggingface-cli download Efficient-Large-Model/coyo-25m-recap --local-dir coyo-25m-recap --repo-type dataset --local-dir-use-symlinks False --resume-download 
JOBS_LIMIT=${1:-8}  # Set your limit here
# model_id=NousResearch/Llama-2-13b-chat-hf
# model_id=google/gemma-7b-it
task="rephrase"

for f in captioner/coyo-25m-recap/*.json; do
  while [ $(jobs -rp | wc -l) -ge $JOBS_LIMIT ]; do
    sleep 1
  done

  model_id="mistralai/Mistral-7B-Instruct-v0.2"
  fname=$(echo $f | rev | cut -d "/" -f 1 | rev)
  model=$(echo $model_id | cut -d "/" -f 2)
  # Replace this with your actual command
  echo "Processing $task on $f and running jobs $(jobs -rp | wc -l)"; \
  srun --label -A nvr_elm_llm -N 1 \
    -p polar3,polar2,polar,batch_block1,grizzly,grizzly2,batch_block2,batch_block3 \
    -t 4:00:00 \
    -J vila:cap2qa-$fname-$model --gpus-per-node 8 --exclusive \
    -e slurm-logs/dev-$task/$fname-$model-$j.err \
    -o slurm-logs/dev-$task/$fname-$model-$j.out \
    torchrun --nproc-per-node 8 llava/data_aug/caption2qa.py --data_path=$f --task=$task --model_id=$model_id &

  model_id="deepseek-ai/deepseek-llm-67b-chat"
  fname=$(echo $f | rev | cut -d "/" -f 1 | rev)
  model=$(echo $model_id | cut -d "/" -f 2)
  # Replace this with your actual command
  echo "Processing $task on $f and running jobs $(jobs -rp | wc -l)"; \
  srun --label -A nvr_elm_llm -N 1 \
    -p polar3,polar2,polar,batch_block1,grizzly,grizzly2,batch_block2,batch_block3 \
    -t 4:00:00 \
    -J vila:cap2qa-$fname-$model --gpus-per-node 8 --exclusive \
    -e slurm-logs/dev-$task/$fname-$model-$j.err \
    -o slurm-logs/dev-$task/$fname-$model-$j.out \
    torchrun --nproc-per-node 8 llava/data_aug/caption2qa.py --data_path=$f --task=$task --model_id=$model_id --load_in_4bit=True &

done
wait
