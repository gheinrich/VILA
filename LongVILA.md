<p align="center">
  <img src="demo_images/longvila-logo.png" width="60%"/>
</p>

# LongVILA: Scaling Long-Context Visual Language Models for Long Videos

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](CODE_LICENSE)
[![Model License](https://img.shields.io/badge/MODEL%20License-CC%20By%20NC%204.0-red.svg)](MODEL_LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

[![Paper](https://img.shields.io/badge/Paper-Arvix%20Link-green)](https://arxiv.org/abs/2408.10188)
[![Huggingface Models](https://img.shields.io/badge/Models-Huggingface%20Models-bron)](https://huggingface.co/collections/Efficient-Large-Model/longvila-66c3fce79284c8209f119b32)

## 💡 Introduction

Long-context capability is critical for multi-modal foundation models. We introduce LongVILA, a full-stack solution for long-context vision-language models, including system, model training, and dataset development. On the system side, we introduce the first long-context Multi-Modal Sequence Parallelism (MM-SP) system that enables long training and inference, enabling 2M context length training on 256 GPUs. MM-SP is also efficient, being 2.1x - 5.7x faster than Ring-Style Sequence Parallelism and 1.1x - 1.4x faster than Megatron-LM in text-only settings. Moreover, it seamlessly integrates with Hugging Face Transformers. For model training, we propose a five-stage pipeline comprising alignment, pre-training, short supervised fine-tuning, context extension, and long supervised fine-tuning. Regarding datasets, we meticulously construct large-scale visual language pre-training datasets and long video instruction-following datasets to support our multi-stage training process. The full-stack solution extends the feasible frame number of VILA by a factor of 128 (from 8 to 1024 frames) and improves long video captioning score from 2.00 to 3.26 (1.6x), achieving 99.5% accuracy in 1400-frames video (274k context length) needle in a haystack. LongVILA-8B also demonstrates consistent accuracy improvements on long videos in the VideoMME benchmark as the video frames increase.

<p align="center">
  <img src="demo_images/LongVILA-pipeline.png" width="100%"/>
</p>

## Installation

```bash
./environment_setup.sh vila
```

## Evaluations

Please refer to `scripts/v1_5/eval/needle.sh`, `scripts/v1_5/eval/video_chatgpt/run_vila_benchmark.sh`, and `llava/eval/video_mme/eval.sh` for needle-in-a-haystack, LongVILA-Caption, and Video MME evaluations.

> \[!Note\]
> 💡**Sequence Parallelism Configuration**
>
> To enable sequence parallelism, you can set the following parameters in the training script:
>
> `seq_parallel_size`:The degree of sequence parallelism (SP). SP is disabled by default (value: -1).
>
> `seq_parallel_ring_size`: The communication process group size using optimized Ring Attention approach in SP. Ring Attention approach is disabled by default in SP.
>
> `seq_parallel_ring_type`: Ring Attention implementation. Support \['ring_varlen', 'zigzag_ring_varlen'\] in 2D attention. Only works when *seq_parallel_ring_size* > 1.
>
> Please note that when SP is enabled, we treat each group of seq_parallel_size GPUs as a single device, with the global batch size calculated as the product of the per-device batch size and the data parallelism size.

## 🔒 License

- The code is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.
- The pretrained weights are released under the [CC-BY-NC-SA-4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).
- The service is a research preview intended for non-commercial use only, and is subject to the following licenses and terms:
  - [Model License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA. For LLAMA3-VILA checkpoints terms of use, please refer to the [LLAMA3 License](https://llama.meta.com/llama3/license/) for additional details.
  - [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI
  - [Dataset Licenses](./data_prepare/LICENSE) for each one used during training.

## Citations

```
@article{longvila,
      title={LongVILA: Scaling Long-Context Visual Language Models for Long Videos},
      author={Fuzhao Xue and Yukang Chen and Dacheng Li and Qinghao Hu and Ligeng Zhu and Xiuyu Li and Yunhao Fang and Haotian Tang and Shang Yang and Zhijian Liu and Yihui He and Hongxu Yin and Pavlo Molchanov and Jan Kautz and Linxi Fan and Yuke Zhu and Yao Lu and Song Han},
      year={2024},
      eprint={2408.10188},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work.
- [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA): we borrowed the long video needle in the haystack evaluation script from this repository.
- [LongLoRA](https://github.com/dvlab-research/LongLoRA): we modified the low-rank long-context fine-tuning code from this repository.
- [USP (YunChang)](https://github.com/feifeibear/long-context-attention): we adopted the 2D attention implementation from this repository.
- [DeepSpeed Ulysses](https://github.com/microsoft/DeepSpeed): we adopted the all-to-all implementation from this repository.
- [RingFlashAttention](https://github.com/zhuzilin/ring-flash-attention): we adopted the ring flash attention implementation from this repository.
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT): we borrowed video evaluation script from this repository.
- [MMC4](https://github.com/allenai/mmc4), [COYO-700M](https://github.com/kakaobrain/coyo-dataset), [M3IT](https://huggingface.co/datasets/MMInstruction/M3IT), [OpenORCA/FLAN](https://huggingface.co/datasets/Open-Orca/FLAN), [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V), [WIT](google-research-datasets/wit), [GSM8K-ScRel](https://github.com/OFA-Sys/gsm8k-ScRel/blob/main/data/train_use.jsonl), [VisualGenome](https://visualgenome.org/api/v0/api_home.html), [VCR](https://visualcommonsense.com/download/), [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [Shot2Story](https://github.com/bytedance/Shot2Story/blob/master/DATA.md), [Youcook2](http://youcook2.eecs.umich.edu/), [Vatex](https://eric-xw.github.io/vatex-website/download.html), [ShareGPT-Video](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction), [ShareGPT4o](https://sharegpt4o.github.io/) for providing datasets used in this research.