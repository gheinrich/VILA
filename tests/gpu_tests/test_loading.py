import os
import shutil
import unittest

import shortuuid
import torch
import torch.nn as nn
from llava.model import LlavaConfig, LlavaLlamaForCausalLM
from llava.train.args import ModelArguments
from llava.train.utils import get_checkpoint_path, prepare_vision_tower_config
from llava.unit_test_utils import requires_gpu
from tqdm import tqdm
from transformers import AutoConfig

torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    device = "cuda:0"
else:
    device = "cpu"


class TestModelInitialization(unittest.TestCase):
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

        self.random_path = "/tmp/test-loading-" + shortuuid.uuid()
        print(f"Using temporary directory: {self.random_path} ...")
        if os.path.exists(self.random_path):
            shutil.rmtree(self.random_path)

    def fast_normal_initialized(self, model, std=0.02, num_reinitialize=10):
        num_reinitialized = 0
        modules = [module for module in model.modules()]
        for module in tqdm(modules[::-1], total=len(modules)):
            num_reinitialized += 1
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                # if module.bias is not None:
                #     module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
            else:
                num_reinitialized -= 1
            if num_reinitialized == num_reinitialize:
                break

    def build_vila_model(self, resume_path=""):
        resume_path = get_checkpoint_path(resume_path)
        if resume_path:
            self.model_args.model_name_or_path = resume_path
            config = AutoConfig.from_pretrained(self.self.model_args.model_name_or_path, trust_remote_code=True)
            config.resume_path = resume_path
            model_cls = eval(config.architectures[0])
            torch.set_default_dtype(torch.bfloat16)
        else:
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
        return model

    @requires_gpu
    def test_first_build(self):
        """
        Build model from scratch.
        """
        model = self.build_vila_model(resume_path=self.random_path)
        ## load pretrained model
        from transformers import CLIPVisionModel, LlamaForCausalLM

        vision_tower = CLIPVisionModel.from_pretrained(self.model_args.vision_tower)
        language_model = LlamaForCausalLM.from_pretrained(self.model_args.model_name_or_path)

        first_loading_params = {param_name: param for param_name, param in model.named_parameters()}
        pretrained_params = {param_name: param for param_name, param in vision_tower.named_parameters()}
        pretrained_params.update({param_name: param for param_name, param in language_model.named_parameters()})
        for k, v in pretrained_params.items():
            if k in first_loading_params.keys():
                self.assertAlmostEqual(torch.equal(v.data, first_loading_params[k].data), True)

    @requires_gpu
    def test_resume(self):
        """
        Resume the whole model from checkpoints.
        """

        model = self.build_vila_model()
        self.fast_normal_initialized(model)
        print("Saving random initialized moodel ...")
        try:
            model.save_pretrained(self.random_path)

            loaded_model = self.build_vila_model(self.random_path)
            saved_state_dict = {param_name: param for param_name, param in model.named_parameters()}
            loaded_state_dict = {param_name: param for param_name, param in loaded_model.named_parameters()}
            for k, v in saved_state_dict.items():
                self.assertEqual(torch.equal(v.data, loaded_state_dict[k].data), True)
            shutil.rmtree(self.random_path)
        except:
            shutil.rmtree(self.random_path)


if __name__ == "__main__":
    unittest.main()
