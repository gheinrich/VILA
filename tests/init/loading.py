import os
import os.path as osp
import sys
import unittest

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
)

from llava.model import *
import llava.model.language_model.llava_llama

from llava.unit_test_utils import requires_gpu, requires_lustre


class TestDatasetLoading(unittest.TestCase):
    @requires_lustre()
    def test_loading(self):
        # print(model)
        resume_path = "Efficient-Large-Model/CI-test-7b-new-format"
        config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
        config.resume_path = resume_path
        model_cls = eval(config.architectures[0])
        config.model_dtype = "torch.bfloat16"

        model = model_cls(
            config=config,
        )
        print(model)

if __name__ == "__main__":
    unittest.main()
