import os
import os.path as osp, shutil
import sys
import unittest

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
)

from llava.model import *
import llava.model.language_model.llava_llama
from collections import OrderedDict
from llava.model.utils import get_model_config
from llava.model.language_model.builder import build_llm
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.configuration_llava import LlavaConfig


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

    def test_from_pretrained(self):
        fpath = "Efficient-Large-Model/CI-format-7b-v2"
        model = AutoModel.from_pretrained(fpath)
        check_params(model)
        
    def test_save_and_reload(self):
        fpath = "Efficient-Large-Model/CI-format-7b-v2"
        model = AutoModel.from_pretrained(fpath)
        shutil.rmtree("checkpoints/tmp", ignore_errors=True)
        model.save_pretrained("checkpoints/tmp")
        model = AutoModel.from_pretrained("checkpoints/tmp")
        check_params(model)

if __name__ == "__main__":
    unittest.main()
