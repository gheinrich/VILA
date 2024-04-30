# Run VILA demo on x86_64 machine

## Build TensorRT-LLM
The first step to build TensorRT-LLM is to fetch the sources:
```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
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
cp -r test_vila.py av.png <TensorRT-LLM directory>/examples/multimodal/
```

Once the image is built, the Docker container can be executed using:
```bash
make -C docker release_run
pip install git+https://github.com/bfshi/scaling_on_scales.git
pip install git+https://github.com/huggingface/transformers@v4.36.2
```
## Build TensorRT engine of VILA model

```bash
# Enter the demo folder
cd <VILA-repo>/demo_trt_llm
export TRTLLM_EXAMPLE_ROOT=/app/tensorrt_llm/examples

# clone original VILA repo
# TODO: Change this back
mkdir -p tmp/hf_models/
cp -r /scratch/weimingc/workspace/VILA-Internal ${VILA_PATH}
# export VILA_PATH="tmp/hf_models/VILA"
# git clone https://github.com/Efficient-Large-Model/VILA.git ${VILA_PATH}

# download vila checkpoint
export MODEL_NAME="vila1.5-2.7b"
cp -r /scratch_weiming/models/VILA1.5-2.7b tmp/hf_models/${MODEL_NAME}
# export MODEL_NAME="vila-2.7B"
# git clone https://huggingface.co/Efficient-Large-Model/${MODEL_NAME} tmp/hf_models/${MODEL_NAME}
```
```
# vison feature
For siglip it should be 729*4 for 4 frames

# convert
# Before
hf_config = LlavaConfig.from_pretrained(hf_model).text_config
# After
if hf_config.model_type == "llava_llama":
    hf_config.llm_cfg["architecture"] = hf_config.llm_cfg["architectures"]
    hf_config.llm_cfg["dtype"] = hf_config.llm_cfg["torch_dtype"]
    hf_config = PretrainedConfig.from_dict(hf_config.llm_cfg)

sys.path.append("/app/tensorrt_llm/examples/multimodal/tmp/hf_models/VILA")
from llava.model import *
# register VILA model
# if "vila" in model_dir:
#     sys.path.append(model_dir + "/../VILA")
#     from llava.model import LlavaConfig, LlavaLlamaForCausalLM
#     AutoConfig.register("llava_llama", LlavaConfig)
#     AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
```
1. TensorRT Engine building using `FP16` and inference

Build TensorRT engine for LLaMA part of VILA from HF checkpoint using `FP16`:
```bash
python convert_checkpoint.py \
    --model_dir tmp/hf_models/${MODEL_NAME} \
    --output_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
    --dtype float16

trtllm-build \
    --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16/1-gpu \
    --output_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
    --gemm_plugin float16 \
    --use_fused_mlp \
    --max_batch_size 2 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --max_multimodal_len 4096
```

2. Build TensorRT engines for visual components

```bash
python build_visual_engine.py --model_path tmp/hf_models/${MODEL_NAME} --model_type vila --vila_path ${VILA_PATH}
```

3. Run the example script
```bash
python run.py  \
    --max_new_tokens 100 \
    --hf_model_dir tmp/hf_models/${MODEL_NAME} \
    --visual_engine_dir visual_engines/${MODEL_NAME} \
    --llm_engine_dir trt_engines/${MODEL_NAME}/fp16/1-gpu \
    --image_file=av.png,merlion.png \
    --input_text="<image>\n<image>\n Please elaborate what you see in the images?" \
    --run_profiling
```

4. (Optional) One can also use LLaVA/VILA with other quantization options, like SmoothQuant and INT4 AWQ, that are supported by LLaMA. Instructions in LLaMA README to enable SmoothQuant and INT4 AWQ can be re-used to generate quantized TRT engines for LLM component of LLaVA/VILA.
```bash
python quantize.py \
     --model_dir tmp/hf_models/${MODEL_NAME} \
     --output_dir tmp/trt_models/${MODEL_NAME}/int4_awq/1-gpu \
     --dtype float16 \
     --qformat int4_awq \
     --calib_size 32

 trtllm-build \
     --checkpoint_dir tmp/trt_models/${MODEL_NAME}/int4_awq/1-gpu \
     --output_dir trt_engines/${MODEL_NAME}/int4_awq/1-gpu \
     --gemm_plugin float16 \
     --max_batch_size 1 \
     --max_input_len 2048 \
     --max_output_len 512 \
     --max_multimodal_len 4096
```