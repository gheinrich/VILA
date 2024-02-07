# conda activate llava
# cd ~/workspace/llava-dev
MPDEL_PATH=${1:-"checkpoints/ablation-vicuna-7b-clip336-PT:coyo_25m_recap+mmc4core+sharegpt4v_pretrain-SFT:vflan_sharegpt4v_sft"}

echo $MPDEL_PATH

MODEL_NAME=$(echo $MPDEL_PATH | rev | cut -d "/" -f 1 | rev)
partition=batch_block1,batch_block2,batch_block3,batch_block4
account=llmservice_nlp_fm


EXP=$MODEL_NAME
FOLDER=eval-result/$EXP
mkdir -p $FOLDER

srun -o "${FOLDER}/seed" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:seed --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15/seed.sh  $MPDEL_PATH &
srun -o "${FOLDER}/mme" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:mme --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15/mme.sh  $MPDEL_PATH &
srun -o "${FOLDER}/pope" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:pope --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15/pope.sh  $MPDEL_PATH &
srun -o "${FOLDER}/sqa" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:sqa --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15//scienceqa.sh  $MPDEL_PATH &
srun -o "${FOLDER}/gqa" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:gqa --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15//gqa.sh  $MPDEL_PATH &
srun -o "${FOLDER}/vqa" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:vqa --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15//vqav2.sh  $MPDEL_PATH &
srun -o "${FOLDER}/vizwiz" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:vizwiz --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15//vizwiz.sh  $MPDEL_PATH &
srun -o "${FOLDER}/mmb" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:mmb --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15//mmbench.sh  $MPDEL_PATH &
srun -o "${FOLDER}/mmb_cn" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:mmb_cn --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15//mmbench_cn.sh  $MPDEL_PATH &
srun -o "${FOLDER}/llava_bench" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:llava_bench --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15//llavabench.sh  $MPDEL_PATH &
srun -o "${FOLDER}/mmvet" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:mmvet --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15//mmvet.sh  $MPDEL_PATH &
srun -o "${FOLDER}/text" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:text --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15//textvqa.sh  $MPDEL_PATH &
srun -o "${FOLDER}/mmmu" -p $partition -A $account -N 1 -t 4:00:00 -J $account-eval:mmmu --gpus-per-node 8 --exclusive  bash ./scripts/eval/llava15//mmmu.sh  $MPDEL_PATH &