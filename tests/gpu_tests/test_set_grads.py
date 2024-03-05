import shutil
import os
import shortuuid
from llava.model import LlavaLlamaForCausalLM, LlavaConfig, AutoConfig
from llava.train.args import ModelArguments, TrainingArguments
from llava.model.utils import get_checkpoint_path, prepare_vision_tower_config

import torch, torch.nn as nn
import unittest


torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    device = "cuda:0"
else:
    device = "cpu"


class TestSetGrads(unittest.TestCase):
    def setUp(self):
        # This test is supposed to run on a single GPU
        if torch.cuda.is_available():
            rank = 0
            torch.cuda.set_device(rank)
        torch.set_default_dtype(torch.bfloat16)

        self.model_args = ModelArguments(
            model_name_or_path="liuhaotian/llava-v1.5-7b",
            vision_tower="openai/clip-vit-large-patch14-336",
            mm_projector_type="mlp2x_gelu",
        )

        self.training_args = TrainingArguments(
            tune_language_model=False, tune_vision_tower=False, tune_mm_projector=False
        )

    def build_vila_model(self):
        ## first time training
        config = LlavaConfig.from_pretrained(self.model_args.model_name_or_path)
        config._attn_implementation = "flash_attention_2"
        torch.set_default_dtype(torch.bfloat16)
        model_cls = LlavaLlamaForCausalLM

        prepare_vision_tower_config(config, self.model_args)
        model = model_cls.from_pretrained(
            self.model_args.model_name_or_path,
            config=config,
        )
        model.get_model().requires_grad_(self.training_args.tune_language_model)
        model.get_model().get_vision_tower().requires_grad_(
            self.training_args.tune_vision_tower
        )
        model.get_model().get_mm_projector().requires_grad_(
            self.training_args.tune_mm_projector
        )
        return model

    def test_tune_projector_and_language_model(self):
        self.training_args.tune_mm_projector = True
        self.training_args.tune_language_model = True
        model = self.build_vila_model()
        for _, param in model.get_model().named_parameters():
            self.assertEqual(param.weight.requries_grad, True)
        for _, param in model.get_model().get_vision_tower().named_parameters():
            self.assertEqual(param.weight.requries_grad, False)
        for _, param in model.get_model().get_mm_projector.named_parameters():
            self.assertEqual(param.weight.requries_grad, True)

    def test_tune_projector_and_vision_tower(self):
        self.training_args.tune_mm_projector = True
        self.training_args.tune_vision_tower = True
        model = self.build_vila_model()
        for _, param in model.get_model().named_parameters():
            self.assertEqual(param.weight.requries_grad, False)
        for _, param in model.get_model().get_vision_tower().named_parameters():
            self.assertEqual(param.weight.requries_grad, True)
        for _, param in model.get_model().get_mm_projector.named_parameters():
            self.assertEqual(param.weight.requries_grad, True)

    def test_tune_projector(self):
        self.training_args.tune_mm_projector = True
        model = self.build_vila_model()
        for _, param in model.get_model().named_parameters():
            self.assertEqual(param.weight.requries_grad, False)
        for _, param in model.get_model().get_vision_tower().named_parameters():
            self.assertEqual(param.weight.requries_grad, False)
        for _, param in model.get_model().get_mm_projector.named_parameters():
            self.assertEqual(param.weight.requries_grad, True)

    def test_tune_vision_tower(self):
        self.training_args.tune_mm_projector = True
        model = self.build_vila_model()
        for _, param in model.get_model().named_parameters():
            self.assertEqual(param.weight.requries_grad, False)
        for _, param in model.get_model().get_vision_tower().named_parameters():
            self.assertEqual(param.weight.requries_grad, True)
        for _, param in model.get_model().get_mm_projector.named_parameters():
            self.assertEqual(param.weight.requries_grad, False)


if __name__ == "__main__":
    unittest.main()
