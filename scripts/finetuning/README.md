# VILA Fine-tuning

## Train

We provide scripts for fine-tuning VILA with full-parameter, LoRA and DoRA methods. The training scipts are under `./scripts/finetuning/train`. We support different combination of frozen, full-parameter and lora modules. For example, to apply lora on LLM and full fine-tune the vision tower, run:

```
bash ./scripts/finetuning/train/no_sqa_lora.sh --vt ft --llm lora --output_dir $OUTPUT_DIR
```

To enable DoRA, add `--use_dora` in `./scripts/finetuning/train/no_sqa_lora.sh`.

## Eval

For evaluation, there are several example scripts under `./scripts/finetuning/eval`. For module fine-tuned with LoRA, use scripts ending with `_lora` and specify both the base model for `MODEL_BASE` and use the `OUTPUT_DIR` in training for `MODEL_PATH` (which stored the adapter weights and non-adapter states).

## Results (mm_projector is always FFT)

| Task      | Metric       | FT       | FT-LLM | FT-VT | FT-VT-LoRA-LLM | FT-VT-DoRA-LLM | LoRA-VT-LoRA-LLM |
|-----------|--------------|----------|--------|-------|----------------|----------------|------------------|
| ScienceQA | Accuracy     | 86.70    | 87.81  | 86.51 |     84.11      |     84.13      | 84.41            |
|           | IMG-Accuracy | 82.45    | 83.39  | 87.11 |     80.47      |     80.52      | 81.06            |
| RefCOCO   | val          | 90.40    | 87.26  | 86.91 |     58.30      |     57.62      | 60.13            |
|           | testA        | 93.41    | 91.16  | 90.83 |     66.78      |     65.60      | 68.25            |
|           | testB        | 86.10    | 81.75  | 83.20 |     51.25      |     50.19      | 52.86            |
| RefCOCO+  | val          | 84.72    | 81.39  | 79.90 |     52.75      |     52.50      | 54.82            |
|           | testA        | 89.35    | 86.76  | 86.40 |     61.96      |     60.95      | 63.76            |
|           | testB        | 77.36    | 72.96  | 72.20 |     43.51      |     43.10      | 45.39            |
| RefCOCOg  | val          | 86.29    | 84.31  | 82.97 |     53.98      |     54.04      | 54.11            |
|           | test         | 86.95    | 84.88  | 83.19 |     55.35      |     54.69      | 54.66            |
