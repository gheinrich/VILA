# Run VILA demo on x86_64 machine

## Build TensorRT-LLM
The first step to build TensorRT-LLM is to fetch the sources:
```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git checkout c89653021e66ca78c55f02b366f404455bc12e8d
git submodule update --init --recursive
git lfs pull
```
Create a TensorRT-LLM Docker image and approximate disk space required to build the image is 63 GB:
```bash
make -C docker release_build
```
Before starting the docker container, mount inference scripts and test image in the docker container as preparation:
```bash
cd <this demo folder>
cp -r llava.py test_vila.py av.png <TensorRT-LLM directory>/examples/multimodal/
```

Once the image is built, the Docker container can be executed using:
```bash
make -C docker release_run
```
## Build TensorRT engine of VILA model
In the docker container, checkout `examples/multimodal/` and Download Huggingface model weights:
```bash
cd examples/multimodal/
export MODEL_NAME="vila-7B"
git clone https://huggingface.co/Efficient-Large-Model/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
```
Make modifications to `tmp/hf_models/${MODEL_NAME}/config.json`, change
```bash
"model_type": "llava",
```
to
```bash
"model_type": "llama",
```
1. TensorRT Engine building using `FP16` and inference

Build TensorRT engine for LLaMA part of VILA from HF checkpoint using `FP16`:
```bash
python3 ../llama/build.py \
    --model_dir tmp/hf_models/${MODEL_NAME} \
    --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
    --dtype float16 \
    --remove_input_padding \
    --use_gpt_attention_plugin float16 \
    --enable_context_fmha \
    --use_gemm_plugin float16 \
    --max_batch_size 1 \
    --max_prompt_embedding_table_size 576

```
Run the inference script for demostration:
```bash
python3 test_vila.py \
    --max_new_tokens 100 \
    --input_text "Please describe the traffic condition." \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
    --decoder_llm \
    --image-file "av.png"
```
2. TensorRT Engine building using `INT4 AWQ` and inference

Weight quantization:
```bash
python3 ../quantization/quantize.py
    --model_dir tmp/hf_models/${MODEL_NAME} \
    --dtype float16 \
    --qformat int4_awq \
    --export_path ./quantized_int4-awq \
    --calib_size 32 \
    --quantize_lm_head
```
Build TensorRT engine for LLaMA part of VILA from HF checkpoint using `INT4 AWQ`:
```bash
python ../llama/build.py --model_dir tmp/hf_models/${MODEL_NAME}
    --quant_ckpt_path ./quantized_int4-awq/llama_tp1_rank0.npz
    --dtype float16
    --remove_input_padding
    --use_gpt_attention_plugin float16
    --enable_context_fmha
    --use_gemm_plugin float16
    --use_weight_only
    --weight_only_precision int4_awq
    --per_group
    --quantize_lm_head
    --output_dir trt_engines/${MODEL_NAME}_int4_AWQ/1-gpu/
    --max_batch_size 1
    --max_prompt_embedding_table_size 576
```
Run the inference script for demonstration:
```bash
python test_vila.py \
    --max_new_tokens 100 \
    --input_text "Please describe the traffic condition." \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --llm_engine_dir trt_engines/${MODEL_NAME}_int4_AWQ/1-gpu \
    --decoder_llm \
    --image-file "demo_images/av.png"
```
