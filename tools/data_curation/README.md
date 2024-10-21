# How to curated datasets

## 0. Setup more datasets

```bash
export VILA_DATASETS=nvidia,cs-oci-ord
```

## 1. Build clip datasets for SFT datasets

```bash
bash tools/data_curation/build.sh
```

## 2. Compute clip similarities btw Benchmark and SFT sets

```bash
python tools/data_curation/clip_calc.py
```

## 3. Filter (SFT) datasets and generate selected indexs.

```bash
python tools/data_curation/clip_filter.py

# Set topk percentage to pick
python tools/data_curation/clip_filter.py  --topk_p=40

# Set topk percentage and target benchmark to filter
python tools/data_curation/clip_filter.py  --benchmark_target="mmmu_val" --topk_p=40
```

## 4. Set env vars and load corresponding datasets.

```bash
# previously, the raw dataset
python tests/python-datasets/single_dst.py sharegpt4v_gpt4_100k

# now, the filtered dataset
export VILA_SLICE_FOLDER=/home/ligengz/workspace/dataset-curation/filter_index
python tests/python-datasets/single_dst.py sharegpt4v_gpt4_100k@30

# this will load slice index
#    `/home/ligengz/workspace/dataset-curation/filter_index/30/sharegpt4v_gpt4_100k.json

```

```bash
/home/ligengz/workspace/VILA-dev/data_curation_dev/filter_index/20
filter 2732497 of 6883312: 0.40

/home/ligengz/workspace/VILA-dev/data_curation_dev/filter_index/30
filter 1829303 of 6883312: 0.27

/home/ligengz/workspace/VILA-dev/data_curation_dev/filter_index/40
filter 1220564 of 6883312: 0.18
```
