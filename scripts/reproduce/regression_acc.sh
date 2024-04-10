CKPT1=checkpoints/CIs-reproduce-stage1
CKPT2=checkpoints/CIs-reproduce-stage2
CKPT3=checkpoints/CIs-reproduce-stage3
# rm $CKPT1 $CKPT2 $CKPT3
wait

ALIGN_DATASET=llava_1_5_mm_align    bash scripts/reproduce/1_mm_align.sh $CKPT1
PT_DATASET=sharegpt4v_pretrain      bash scripts/reproduce/2_pretrain.sh $CKPT1 $CKPT2
SFT_DATASET=sharegpt4v_sft          bash scripts/reproduce/3_sft_captioner.sh $CKPT2 $CKPT3

'''
srun -A $SLURM_ACCOUNT \
    -N 8 \
    -p $SLURM_PARTITION -t 4:00:00 \
    -J vila:CI-acc-regression \
    --gpus-per-node 8 --exclusive \
    bash scripts/reproduce/regression_acc.sh
'''