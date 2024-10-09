import torch
import transformer_engine.pytorch as te
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llava.model import QLlamaConfig
from llava.model.language_model.qllama import QLlamaDecoderLayer

# QLlama
llm_config = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "model_max_length": 4096,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "tokenizer_model_max_length": 4096,
    "tokenizer_padding_side": "right",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.37.2",
    "use_cache": True,
    "vocab_size": 32000,
    "quantize_model": "te_linear",
    "_pre_quantization_dtype": torch.bfloat16,
    "fabit": "E4M3",
    "fwbit": "E4M3",
    "bobit": "E5M2",
    "row_blocksize": -1,
    "col_blocksize": -1,
}

llm_config = QLlamaConfig(**llm_config)
model_layer = QLlamaDecoderLayer(llm_config, layer_idx=0).cuda().to(torch.bfloat16)

# # Llama
# llm_config = {
#   "architectures": [
#     "LlamaForCausalLM"
#   ],
#   "attention_bias": False,
#   "attention_dropout": 0.0,
#   "bos_token_id": 1,
#   "eos_token_id": 2,
#   "hidden_act": "silu",
#   "hidden_size": 4096,
#   "initializer_range": 0.02,
#   "intermediate_size": 11008,
#   "max_position_embeddings": 4096,
#   "model_max_length": 4096,
#   "model_type": "llama",
#   "num_attention_heads": 32,
#   "num_hidden_layers": 32,
#   "num_key_value_heads": 32,
#   "pad_token_id": 0,
#   "pretraining_tp": 1,
#   "rms_norm_eps": 1e-05,
#   "rope_scaling": None,
#   "rope_theta": 10000.0,
#   "tie_word_embeddings": False,
#   "tokenizer_model_max_length": 4096,
#   "tokenizer_padding_side": "right",
#   "torch_dtype": "bfloat16",
#   "transformers_version": "4.37.2",
#   "use_cache": True,
#   "vocab_size": 32000,
# }

# llm_config = LlamaConfig(**llm_config)
# model_layer = LlamaDecoderLayer(llm_config).cuda().to(torch.bfloat16)

torch.manual_seed(0)
batch_sizes = [1, 4, 8, 16, 32]
sequence_lengths = [1024, 2048]
# batch_sizes = [16]
# sequence_lengths = [2048]
n_repeat = 15

# Forward Time Benchmark
for batch_size in batch_sizes:
    for sequence_length in sequence_lengths:
        dummy_input = torch.rand((batch_size, sequence_length, llm_config.hidden_size), dtype=torch.bfloat16).cuda()
        dummy_attention_mask = torch.ones((batch_size, sequence_length)).cuda() == 1  # All True Matrix
        dummy_position_ids = torch.tensor([range(sequence_length) for _ in range(batch_size)]).cuda()

        start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

        # with te.fp8_autocast(enabled=True):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for i in range(n_repeat):
                start_event[i].record()
                dummy_output = model_layer(dummy_input, dummy_attention_mask, dummy_position_ids)
                end_event[i].record()
            torch.cuda.synchronize()

        times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        avg_time = torch.median(times)

        print(times)
        print(f"Batch Size: {batch_size} | Sequence_length: {sequence_length} | Time: {avg_time}")

# Backward Time Benchmark
for batch_size in batch_sizes:
    for sequence_length in sequence_lengths:
        dummy_input = torch.rand((batch_size, sequence_length, llm_config.hidden_size), dtype=torch.bfloat16).cuda()
        dummy_grad = torch.rand((batch_size, sequence_length, llm_config.hidden_size), dtype=torch.bfloat16).cuda()
        dummy_attention_mask = torch.ones((batch_size, sequence_length)).cuda() == 1  # All True Matrix
        dummy_position_ids = torch.tensor([range(sequence_length) for _ in range(batch_size)]).cuda()

        start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
        end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

        # with te.fp8_autocast(enabled=True):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            torch.empty(
                1, device="cuda", requires_grad=True
            ).backward()  # Triton will throw RuntimeError: Triton Error [CUDA]: invalid device context if you comment this line
            for i in range(n_repeat):
                dummy_output = model_layer(dummy_input, dummy_attention_mask, dummy_position_ids)
                start_event[i].record()
                dummy_output[0].backward(dummy_grad, retain_graph=True)
                end_event[i].record()
            torch.cuda.synchronize()

        times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        avg_time = torch.median(times)

        print(f"Batch Size: {batch_size} | Sequence_length: {sequence_length} | Time: {avg_time}")

# # Activation Memory Comsumption Benchmark
# model_memory = torch.cuda.memory_allocated()
# print(f"Model Memory: {model_memory / 1024 ** 2} MB")

# batch_size, sequence_length = 4, 1024
# dummy_input = torch.rand((batch_size, sequence_length, llm_config.hidden_size), dtype=torch.bfloat16).cuda().requires_grad_(True)
# dummy_attention_mask = (torch.ones((batch_size, sequence_length)).cuda() == 1) # All True Matrix
# dummy_position_ids = torch.tensor([range(sequence_length) for _ in range(batch_size)]).cuda()

# # with te.fp8_autocast(enabled=True):
# with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#     dummy_output = model_layer(dummy_input, dummy_attention_mask, dummy_position_ids)
#     loss = dummy_output[0].mean()

# loss.backward()
# del dummy_input, dummy_attention_mask, dummy_position_ids, dummy_output

# for batch_size in batch_sizes:
#     for sequence_length in sequence_lengths:
#         for _ in range(n_repeat):
#             dummy_input = torch.rand((batch_size, sequence_length, llm_config.hidden_size), dtype=torch.bfloat16).cuda().requires_grad_(True)
#             dummy_attention_mask = (torch.ones((batch_size, sequence_length)).cuda() == 1) # All True Matrix
#             dummy_position_ids = torch.tensor([range(sequence_length) for _ in range(batch_size)]).cuda()

#             torch.cuda.empty_cache()
#             start_memory = torch.cuda.memory_allocated()
#             # with te.fp8_autocast(enabled=True):
#             with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#                 dummy_output = model_layer(dummy_input, dummy_attention_mask, dummy_position_ids)
#                 loss = dummy_output[0].mean()
#             end_memory = torch.cuda.memory_allocated()
#             loss.backward()
#             del dummy_input, dummy_attention_mask, dummy_position_ids, dummy_output

#             print(f"Batch Size: {batch_size} | Sequence_length: {sequence_length} | Activation Memory: {(end_memory - start_memory) / 1024 ** 2} MB | "
#                 f"Start Memory {start_memory / 1024 ** 2} MB | End Memory {end_memory / 1024 ** 2} MB")
