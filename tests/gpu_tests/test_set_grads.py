import unittest
import torch
import copy

from transformers import AutoTokenizer
from llava.model import LlavaConfig, LlavaLlamaForCausalLM
from llava.train.args import ModelArguments, TrainingArguments
from llava.train.utils import prepare_vision_tower_config
from llava.unit_test_utils import requires_gpu

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
            tune_language_model=False,
            tune_vision_tower=False,
            tune_mm_projector=False,
            output_dir=None,
        )

    def single_forward(self):
        print("Preprocessing inputs...")
        data = copy.deepcopy(self.data)
        data["input_ids"] = data["input_ids"].to(device)
        data["images"] = data["images"].to(torch.bfloat16).to(device)
        data["attention_mask"] = data["attention_mask"].to(device)
        data["labels"] = data["labels"].to(device)
        data["position_ids"] = None
        data["past_key_values"] = None

        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.model.prepare_inputs_labels_for_multimodal(**data)

        print("Packing inputs...")
        (
            _,
            new_position_ids,
            new_attention_mask,
            _,
            new_inputs_embeds,
            new_labels,
            sorted_seqlens_in_batch,
        ) = self.model.repack_multimodal_data(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        )

        print("Running models...")

        # forward results with input packing
        outputs = super(LlavaLlamaForCausalLM, self.model).forward(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=None,
            inputs_embeds=new_inputs_embeds,
            labels=new_labels,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            seqlens_in_batch=sorted_seqlens_in_batch,
        )
        loss = outputs.loss
        loss.backward()

    def build_vila_model(self):
        ## first time training
        config = LlavaConfig.from_pretrained(self.model_args.model_name_or_path)
        config._attn_implementation = "flash_attention_2"
        torch.set_default_dtype(torch.bfloat16)
        model_cls = LlavaLlamaForCausalLM

        print("Initializing data...")
        data = torch.load("tests/sample_data/test_packing.pth")
        self.data = data
        print("Initializing model...")
        prepare_vision_tower_config(config, self.model_args)
        self.model = model_cls.from_pretrained(
            self.model_args.model_name_or_path,
            config=config,
        ).to(torch.bfloat16).to(device)
        print("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            model_max_length=4096,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )
        # necessary for model forward
        self.model.pad_token_id = self.tokenizer.pad_token_id
        self.model.get_model().requires_grad_(self.training_args.tune_language_model)
        self.model.get_model().get_vision_tower().requires_grad_(
            self.training_args.tune_vision_tower
        )
        self.model.get_model().get_mm_projector().requires_grad_(
            self.training_args.tune_mm_projector
        )

    def verify_grads_state(
        self, model, tune_language_model, tune_vision_tower, tune_mm_projector
    ):
        print("Checking gradients...")
        for param_name, param in model.get_model().named_parameters():
            if "vision_tower" not in param_name and "mm_projector" not in param_name:
                self.assertEqual(param.grad is not None, tune_language_model)
        for name, param in model.get_model().get_vision_tower().named_parameters():
            # some models do not use the last layer feature.
            if "post_layernorm" in name:
                continue
            self.assertEqual(param.grad is not None, tune_vision_tower)
        for _, param in model.get_model().get_mm_projector().named_parameters():
            self.assertEqual(param.grad is not None, tune_mm_projector)

    # Actually can write a loop to iterate through combinations
    @requires_gpu
    def test_tune_projector_and_language_model(self):
        print("Testing tune projector and language model ...")
        self.training_args.tune_mm_projector = True
        self.training_args.tune_language_model = True
        self.build_vila_model()
        self.single_forward()

        self.verify_grads_state(
            self.model,
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )

    @requires_gpu
    def test_tune_projector_and_vision_tower(self):
        print("Testing tune projector and vision tower ...")
        self.training_args.tune_mm_projector = True
        self.training_args.tune_vision_tower = True
        self.build_vila_model()
        self.single_forward()
        
        self.verify_grads_state(
            self.model,
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )

    @requires_gpu
    def test_tune_projector(self):
        print("Testing tune projector ...")
        self.training_args.tune_mm_projector = True
        self.build_vila_model()
        self.single_forward()
        
        self.verify_grads_state(
            self.model,
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )

    @requires_gpu
    def test_tune_vision_tower(self):
        print("Testing tune vision tower ...")
        self.training_args.tune_vision_tower = True
        self.build_vila_model()
        self.single_forward()
        
        self.verify_grads_state(
            self.model,
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )

    @requires_gpu
    def test_tune_language_model(self):
        print("Testing tune language model ...")
        self.training_args.tune_language_model = True
        self.build_vila_model()
        self.single_forward()
        
        self.verify_grads_state(
            self.model,
            self.training_args.tune_language_model,
            self.training_args.tune_vision_tower,
            self.training_args.tune_mm_projector,
        )


if __name__ == "__main__":
    unittest.main()
