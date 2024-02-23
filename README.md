# 🌋 VILA: On Pre-training for Visual Language Models
[arxiv](https://arxiv.org/abs/2312.07533) / [demo](https://vila-demo.hanlab.ai/) 

## 💡 News
- [2024/02] We release quantized VILA models! The quantization is done through AWQ (can be used w/ [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat) and [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine)) and TensorRT-LLM (also AWQ inside, compatible w/ TensorRT-LLM runtime, see [README](demo_trt_llm/README.md) for more details). TinyChatEngine can run quantized VILA model on different platforms, including x86 and Arm. It runs VILA-7B at **36 tokesn on the Jetson Orin** and **11 tokens/s on the Apple M1 MacBook Pro**.
- [2024/02] VILA is released! We propose interlevaed pretraining to improve large visual language models. VILA comes with impressive in-context learning capabilities and demonstrates higher benchmarks. We re-wrote the entire codebase based on LLava1.5 and open source everything! (including training code, evaluation code, datasets, model ckpts, etc.) Compared to original LLava1.5 codebase, we support more flexible data mixing w/ different data format and high efficient training through example packing.
- [2023/12] [Paper](https://arxiv.org/abs/2312.07533) is on Arxiv!

## Performance

| model | VQAv2 (server-dev) | GQA | VizWiz (server-dev) | SQA-I | VQA-T | POPE | MME | MMB (server) | MMB-CN (server) | SEED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vila-7b | 80.3 | 63.1 | 59.6 | 68 | 62.6 | 86.3 | 1489.38 | 69.8 | 61 | 61.7 |
| vila-7b-INT4-AWQ | 80.1 | 63.0 | 57.8 | 68 | 61.9 | 85.3 | 1486.25 | 68.8 | 59 | 61.3 |
| vila-13b | 80.5 | 63.6 | 63.1 | 70.5 | 64 | 86.3 | 1553.6 | 73.8 | 66.7 | 62.8 |
| vila-13b-INT4-AWQ | 80.4 | 63.6 | 63.0 | 71.2 | 63.5 | 86.98 | 1552.9 | 73.6 | 66.3 | 62.2 |

### Inference speed ( Token/sec )

| model | backend | A100 | 4090 | Orin |
| --- | --- | --- | --- | --- |
| vila-7b | ||| 11.5 |
| vila-7b-INT4-AWQ | TinyChat ||| 35.6 |
| vila-7b-INT4-AWQ | TRT-LLM ||||
| vila-13b | ||| 6.1 |
| vila-13b-INT4-AWQ | TinyChat ||| 17.5 |
| vila-13b-INT4-AWQ | TRT-LLM |||  |


## VILA Examples

https://github.com/Efficient-Large-Model/VILA-OSS/assets/7783214/a3eed8da-af13-4eb9-8030-383b2a4562d6

<details>
<summary>More examples</summary>

https://github.com/Efficient-Large-Model/VILA-OSS/assets/7783214/ab84f190-00c6-404a-9d74-23936145f5e6

https://github.com/Efficient-Large-Model/VILA-OSS/assets/7783214/2c8f3d0d-b319-432e-ae95-b9e123c00789

https://github.com/Efficient-Large-Model/VILA-OSS/assets/7783214/8d78a86f-417e-43d4-8b37-a54dec3ac0fa

</details>

## Installation

```bash
./environment_setup.sh
```

or follow the instructions below in order.

```
conda create -n vila python=3.10 -y
conda activate vila

pip install --upgrade pip  # enable PEP 660 support
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.4.2/flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.4.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e .
pip install -e ".[train]"

pip install git+https://github.com/huggingface/transformers@v4.38.1
cp -r ./llava/train/transformers_replace/* ~/anaconda3/envs/vila/lib/python3.10/site-packages/transformers/models/
```

## Training 

VILA training contains three steps

### Step-1: Alignment
We utilize LLaVA-CC3M-Pretrain-595K dataset to align the textual and visual modalities.

The stage 1 script takes in two parameters and it can run on a single 8xA100 node. `BASE_MODEL_PATH` points to a online or local huggingface repository, such as lmsys/vicuna-7b-v1.5. `OUTPUT_NAME` points to a target directory under `checkpoints`, which will save the trained multimodal projector afterwards.

```bash
bash scripts/v1_5/paper/1_mm_align.sh [BASE_MODEL_PATH] [OUTPUT_NAME]
```

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| VILA-7B | 256 | 2e-5 | 1 | 4096 | 0 |
| VILA-13B | 256 | 2e-5 | 1 | 4096 | 0 |


### Step-2: Pretraining
use MMC4 and Coyo dataset to train vLLMs with interleaved image-text pairs.
    
```bash
bash scripts/v1_5/paper/2_pretrain_mmc4_coyo.sh [CODE_PATH] [BASE_MODEL_PATH] [STAGE1_PATH] [OUTPUT_NAME]
```

The stage 2 script takes in four arguments. `CODE_PATH` is the absolute path to our VILA codebase, `BASE_MODEL_PATH` has similar meaning to what is presented in the stage 1 script. `STAGE1_PATH` points to the `OUTPUT_NAME` of stage 1 (i.e. where the stage 1 checkpoint is stored). `OUTPUT_NAME` is the desired folder name under `checkpoints` that saves the pretraining checkpoint. The script we provided for this stage is executed on slurm, and we expect it to execute on 16 nodes (128 GPUs). 

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| VILA-7B | 1024 | 5e-5 | 1 | 4096 | 0 |
| VILA-13B | 1024 | 5e-5 | 1 | 4096 | 0 |

### Step-3: Supervised fine-tuning 
This is the last stage of VILA training, in which we tune the model to follow multimodal instructions on a subset of M3IT, FLAN and ShareGPT4V. This stage runs on a 8xA100 node.

```bash
bash scripts/v1_5/paper/3_sft.sh [STAGE2_PATH] [OUTPUT_NAME]
```
The stage 3 script takes in two arguments. `STAGE2_PATH` points to the `OUTPUT_NAME` of the stage 2 script (i.e. where the stage 2 checkpoint is stored). `OUTPUT_NAME` is the desired folder name under `checkpoints` that stores the final checkpoint.

| Hyperparameter | Global Batch Size | Learning rate | Epochs | Max length | Weight decay |
| --- | ---: | ---: | ---: | ---: | ---: |
| VILA-7B | 128 | 2e-5 | 1 | 4096 | 0 |
| VILA-13B | 128 | 2e-5 | 1 | 4096 | 0 |

To train with fewer GPUs/nodes, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly.  As long as the global batch size same (`per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`) are kept the same, the training precision will not be affected.

Stage 1 completes within 3.5 (7B) - 5.5 (13B) hours on 8xA100, Stage 2 completes within 30 hours on 128xA100 for VILA-7b, and stage 3 completes in 25 (7B) - 40 (13B) hours on 8xA100.

See [scripts/data_prep/README.md](scripts/data_prep/README.md) for more information about how to prepare datasets.

## Evaluations

You can follow [Llava1.5 eval](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) to download all datasets. After downloading all datasets, please put them under `playground/data/eval`. 

We provide a push-the-button script to perform evaluation on all 10 datasets that do not require GPT-assisted evaluation:

```bash
./scripts/v1_5/eval/eval_all.sh [CHECKPOINT_PATH] [MODEL_NAME]
```

This script takes in two parameters, `CHECKPOINT_PATH` points to the stage 3 model checkpoint, and `MODEL_NAME` will be the name of evaluation results. 


[VQAv2](https://eval.ai/web/challenges/challenge-page/830/my-submission) and [Vizwiz](https://eval.ai/web/challenges/challenge-page/2185/my-submission) evaluations are hosted on eval.ai. You need to register an account and create a team to be able to submit eval.

MMBench and MMBench_CN eval are hosted on another [evaluation server](https://opencompass.org.cn/leaderboard-multimodal). Make sure you change the name of the file before submitting, otherwise the server caches results and will always return wrong result to you.

We provide a quick script to automatically organize the prediction files that need to be submitted to servers:

```bash
python scripts/v1_5/eval/copy_predictions.py [MODEL_NAME]
```

You will be able to find the predictions under `playground/data/predictions_upload/[MODEL_NAME]` after executing this script.

## Inference

We provide snippets for quick inference with user prompts and images.

VILA-7B inference:
```bash
cd VILA-OSS
python -W ignore llava/eval/run_llava.py \
    --model-name Efficient-Large-Model/VILA-7b \
    --conv-mode vicuna_v1 \
    --query "<image>\n Please describe the traffic condition." \
    --image-file "demo_trt_llm/av.png"
```

VILA-13B inference:
```bash
cd VILA-OSS
python -W ignore llava/eval/run_llava.py \
    --model-name Efficient-Large-Model/VILA-13b \
    --conv-mode vicuna_v1 \
    --query "<image>\n Please describe the traffic condition." \
    --image-file "demo_trt_llm/av.png"
```

## 🔒 License
- The code is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.
- The pretrained weights are released under the [CC-BY-NC-SA-4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).
- The service is a research preview intended for non-commercial use only, and is subject to the following licenses and terms:
    - [Model License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA
    - [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI
    - [Dataset Licenses](./data/LICENSE) for each one used during training.

## Team
| | | |  
| --- | --- | ---| 
[*Ji Lin](https://www.linji.me/): OpenAI (work done at Nvidia and MIT) |  [*Hongxu Yin](https://hongxu-yin.github.io/): Nvidia |  [*Yao Lu](https://scholar.google.com/citations?user=OI7zFmwAAAAJ&hl=en): Nvidia 
[Huizi Mao](https://scholar.google.com/citations?user=r5WezOYAAAAJ&hl=zh-CN): Nvidia |  [Wei Ping](https://scholar.google.com/citations?user=6gKEYRgAAAAJ&hl=en): Nvidia |   [Pavlo Molchanov](https://www.pmolchanov.com/): Nvidia |  
[Andrew Tao](https://scholar.google.com/citations?user=Wel9l1wAAAAJ&hl=en): Nvidia |  [Jan Kautz](https://scholar.google.com/citations?user=P9FclNEAAAAJ&hl=en): Nvidia  |   [Mohammad Shoeybi](https://scholar.google.com/citations?user=62ElavIAAAAJ&hl=en): Nvidia |  
[Haotian Tang](http://kentang.net/): MIT |  [Shang Yang](https://ys-2020.github.io/): MIT |  [Ligeng Zhu](https://lzhu.me/): Nvidia, MIT | 
[Wei-Chen Wang](https://scholar.google.com/citations?user=eYrx3KAAAAAJ&hl=en): MIT |  [Fuzhao Xue](https://xuefuzhao.github.io/): Nvidia, National University of Singapore |  [Yunhao Fang](https://seerkfang.github.io/): Nvidia, University of California, San Diego |  
[Yukang Chen](https://yukangchen.com/): Nvidia, The Chinese University of Hong Kong |  [Yue Shen](https://www.linkedin.com/in/yue-james-shen/): Nvidia |  [Song Han](http://songhan.mit.edu/): Nvidia, MIT


## Citations

```
@misc{lin2023vila,
      title={VILA: On Pre-training for Visual Language Models}, 
      author={Ji Lin and Hongxu Yin and Wei Ping and Yao Lu and Pavlo Molchanov and Andrew Tao and Huizi Mao and Jan Kautz and Mohammad Shoeybi and Song Han},
      year={2023},
      eprint={2312.07533},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work.
- [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT): we borrowed video evaluation script from this repository.
- [MMC4](https://github.com/allenai/mmc4), [COYO-700M](https://github.com/kakaobrain/coyo-dataset), [M3IT](https://huggingface.co/datasets/MMInstruction/M3IT), [OpenORCA/FLAN](https://huggingface.co/datasets/Open-Orca/FLAN), [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V) for providing datasets used in this research.
