set -e

export CUDA_LAUNCH_BLOCKING=1

files=(
    sharegpt4v_gpt4_100k
    llava_instruct
    sharegpt4v_sft
    dvqa_train_200k
    chartqa_train_18k
    ai2d_train_12k
    docvqa_train_10k
    geoqa
    synthdog_en
    scienceqa
    wit_subset
    math
    sherlock
    idefics2_sft
    llave_onevision_images_sft
    cambrian_1375k
    stem_qa
    nv_mm_sft
    k710
    ssv2
    reason_clevrerqa
    reason_clevrermc
    vcg_human
    vflan
    refcoco_train
    shikra
    lrv_instruction
    textocr_qa
    mmc_instruction
    nextqa_mc
    unimm_chat
    svit
    mmbench_val
    cvbench
)

files=(
    idefics2_sft
    cvbench
)

# rm -rfv runs/dev
# export VILA_SLURM_ACCOUNT=llmservice_nlp_fm
export VILA_SLURM_PARTITION=$VILA_SLURM_PARTITION,interactive
for file in ${files[@]}; do
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "[locally] Processing $file ";
        torchrun --nproc-per-node 8 tools/data_curation/build.py \
            --mmdb-dir data_curation_dev/mmdb \
            --dataset $file \
            --batch-size 512 \
            --model-name-or-path google/siglip-so400m-patch14-384 \
            --num-workers 8
    else
        # Set max parallel run concurrency to 10
        while [ $(jobs -p | wc -l) -ge 16 ]; do
            sleep 5
        done
        echo "[submit to slurm] Processing $file ";
        vila-run -m dev --max-retry 1 -J parallel-emb-$file \
            torchrun --nproc-per-node 8 tools/data_curation/build.py \
                --mmdb-dir data_curation_dev/mmdb \
                --dataset $file \
                --batch-size 512 \
                --model-name-or-path google/siglip-so400m-patch14-384 &
    fi
done
wait
exit 0

file=geoqa
torchrun --nproc-per-node 1 tools/data_curation/build.py \
    --mmdb-dir data_curation_dev/mmdb \
    --dataset $file \
    --batch-size 8 \
    --model-name-or-path google/siglip-so400m-patch14-384
