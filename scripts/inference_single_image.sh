source ~/anaconda3/bin/activate
conda init
source ~/.bashrc
conda activate vila
which python

cd /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/multi-modality-research/VILA/

python llava/eval/run_vila.py --model-name ./checkpoints/vicuna-7b-clip336-mmc4sub+coyo-finetune-llava15+vflan+sharegpt4v+video-nosqa-linear-e1004 --image-file /lustre/fsw/portfolios/nvr/projects/nvr_aialgo_robogptagent/loragen_workspace/multi-modality-research/VILA/demo_images/av.png --query "Give a short and clear explanation of the subsequent image." --conv-mode vicuna_v1_1
