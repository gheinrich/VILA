rm -rfv checkpoints/stage1
rm -rfv checkpoints/stage2
rm -rfv checkpoints/stage3
WANDB_DISABLED=false

bash scripts/v1_5/tests/1_mm_align.sh checkpoints/stage1
bash scripts/v1_5/tests/2_pretrain.sh checkpoints/stage1 checkpoints/stage2
bash scripts/v1_5/tests/3_sft_captioner.sh checkpoints/stage2 checkpoints/stage3