import os
import os.path as osp, shutil
import sys
import unittest

import torch
from torch.utils.data import DataLoader

from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    PreTrainedModel,
    AutoConfig,
    AutoModel,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)

from llava.unit_test_utils import requires_gpu, requires_lustre


def check_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 10e5

class TestModelLoadingAndSaving(unittest.TestCase):
    def test_from_config(self):
        fpath = "Efficient-Large-Model/CI-format-7b-v2"
        config = AutoConfig.from_pretrained(fpath)
        model = AutoModel.from_config(config)
        check_params(model)

    def test_from_config(self):
        fpath = "Efficient-Large-Model/CI-format-7b-v2"
        model = AutoModel.from_pretrained(fpath)
        check_params(model)
        
    def test_from_config(self):
        fpath = "Efficient-Large-Model/CI-format-7b-v2"
        model = AutoModel.from_pretrained(fpath)
        shutil.rmtree("checkpoints/tmp", ignore_errors=True)
        model.save_pretrained("checkpoints/tmp")
        model = AutoModel.from_pretrained("checkpoints/tmp")
        check_params(model)

if __name__ == "__main__":
    unittest.main()
