import unittest

import torch
import transformers
from transformers.models.siglip import SiglipImageProcessor

from llava import conversation as conversation_lib
# from llava.train.token_config import (
#     DEFAULT_IMAGE_PATCH_TOKEN,
# )
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN
from llava.data import datasets_mixture
from llava.data.dataset import make_supervised_data_module
from llava.train.args import DataArguments, TrainingArguments
from llava.unit_test_utils import requires_gpu, requires_lustre


def test_make_supervised_data_module(dataset_name, max_samples=100):
    # datasets_mixture.register_datasets_mixtures()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "lmsys/vicuna-7b-v1.5",
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
        legacy=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    data_args = DataArguments(
        data_mixture=dataset_name,
        is_multimodal=True,
        lazy_preprocess=True,
    )
    data_args.image_processor = image_processor
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    training_args = TrainingArguments(
        output_dir="output",
    )

    # training_args["process_index"] = 0
    # training_args.world_size = 1
    data_args.mm_use_im_start_end = False
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    dataset = data_module["train_dataset"]
    index = 0
    dataset_len = len(data_module["train_dataset"])
    from torch.utils.data import DataLoader

    dloader = DataLoader(dataset, batch_size=16, collate_fn=data_module["data_collator"], num_workers=4)
    dloader_len = len(dloader)
    for batch in dloader:
        # if index > min(max_samples, dloader_len):
        #     break
        print(type(batch), batch.keys())
        # print(batch["image"].shape)
        # print(batch["input_ids"].shape[0])
        index += 1
        # print(batch['image'].shape, batch['input_ids'].shape)
    # print(data_module)


class TestStringMethods(unittest.TestCase):
    @requires_lustre()
    def test_dataloader_panda70m(self):
        test_make_supervised_data_module(dataset_name="panda70m")


if __name__ == "__main__":
    unittest.main()
