
# Data Preparation for Training VILA

To train VILA, we used the following datasets:

| Stage                   | Datasets                    |
| ----------------------- | --------------------------- |
| 0. Initialize projector | CCS                         |
| 1. Pre-training         | MMC4-core, COYO-700M subset |
| 2. SFT                  | LLaVA-1.5, VFLAN, ShareGPT, TextFLAN      |

### 0. CCS
We use [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/chat.json) to train the visual language projector

### 1.1. MMC4-core dataset
Due to the limit of compute, we pre-train VILA on the smaller core set of MMC4 instead of the full set. 

1. Firstly, download the annotations of the MMC4-core dataset here: https://github.com/allenai/mmc4. We used the non-fewer-face split, and you may need to request the access [here](https://forms.gle/VYtcNY8aYaUANK9f8). 

2. Now modify the input and output path in `mmc4_downloader.py` and run the following script to scrawl the MMC4 images:
```bash
cd mmc4
python mmc4_downloader.py
```
Note that due to the expiration of image urls, you may end up getting a subset of the entire corpus. 

The scrawling may take a long time. Optionally, you can also shard the workload over multiple jobs/machines concurrently to speed up the process:
```bash
# provide the start and end index of the jsonl shard. There are 23098 - 14 shards totally
python mmc4_downloader.py 0 1000  # worker 1
python mmc4_downloader.py 1000 2000  # worker 2
...
```

3. Filter out invalid samples in MMC4:

```bash
python mmc4_filter_and_counter.py
```

4. Merge images and text into a unified pickle file for each shard:

```bash
python mmc4_merger.py
```

### 1.2 COYO-700M
1. Download the metadata of COYO-700M:
```bash
# download coyo-700m (requires git lfs)
git clone https://huggingface.co/datasets/kakaobrain/coyo-700m
```

2. Scrawl the COYO images. Note that here we only keep a 20% subset in each shard with the highest CLIP similarity, to balance compute budget and data quality. 

There are totally 128 shards of annotations. Now download each one with the script:
```bash
cd coyo
python coyo_downloader.py $SHARD  # $SHARD is 0, 1, ..., 127
```

3. Split downloaded COYO data into multiple shards:
```bash
python coyo_splitter.py
```

### 2.1 LLaVA-1.5 instruction data

We use this [file](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) in our experiments. Please download this dataset from LLaVA authors.

### 2.2 VFlan blend
1. Download FLAN and M3IT datasets:

```bash
git clone https://huggingface.co/datasets/Open-Orca/FLAN
git clone https://huggingface.co/datasets/MMInstruction/M3IT
```

2. Preprocess FLAN dataset (sample 1M data from 378M samples):

```bash
cd sft
python preprocess_flan.py
```

3. Preprocess M3IT dataset:

```bash
python preprocess_m3it.py
```

4. (Optional) Split FLAN+M3IT into multiple chunks to reduce CPU memory pressure during training:

```bash
python split_vflan.py
```

### 2.3 ShareGPT
We also mix in text-only instruction data like ShareGPT data and text-only FLAN data.

1. The ShareGPT data can be obtained [here](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).
2. The text-only FLAN data can be downloaded [here](https://huggingface.co/datasets/Open-Orca/FLAN). Due to the large size of the dataset, we randomly sampled a subset of 1M samples for training.