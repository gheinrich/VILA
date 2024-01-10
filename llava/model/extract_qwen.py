from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/home/jil/models/Qwen-VL", trust_remote_code=True)

print(model.config)
# print(model)

model = model.transformer.visual
model = model.cuda()

print("&" * 200)
print(model.config)

# model.save_pretrained("/home/jil/models/Qwen-VL/visual", save_config=True)

import torch

x = torch.randn(1, 3, 448, 448).cuda()

y = model(x)

print(y.shape)