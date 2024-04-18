import unittest
import torch
from llava.data import dataset
import transformers
from llava.unit_test_utils import requires_lustre
from llava import conversation as conversation_lib


TEST_SOURCE_IMAGE = [[{"from": "human", "value": "<image>\nWho is the author of this book?\nAnswer the question using a single word or phrase."}, {"from": "gpt", "value": "*"}, {"from": "human", "value": "What is the title of this book?"}, {"from": "gpt", "value": "Rough Guide Map of South Africa, Lesotho"}, {"from": "human", "value": "What is the genre of this book?"}, {"from": "gpt", "value": "Travel"}, {"from": "human", "value": "Is this book related to Travel?"}, {"from": "gpt", "value": "Yes"}, {"from": "human", "value": "Is this book related to History?"}, {"from": "gpt", "value": "No"}]]

TEST_SOURCE_TEXT_ONLY = [[{"from": "human", "value": "Who is the author of this book?\nAnswer the question using a single word or phrase."}, {"from": "gpt", "value": "*"}, {"from": "human", "value": "What is the title of this book?"}, {"from": "gpt", "value": "Rough Guide Map of South Africa, Lesotho"}, {"from": "human", "value": "What is the genre of this book?"}, {"from": "gpt", "value": "Travel"}, {"from": "human", "value": "Is this book related to Travel?"}, {"from": "gpt", "value": "Yes"}, {"from": "human", "value": "Is this book related to History?"}, {"from": "gpt", "value": "No"}]]


class TestPromptPreprocess(unittest.TestCase):

    def setUp(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            '/home/jasonlu/models/Nous-Hermes-2-Yi-34B',
            cache_dir='',
            model_max_length=4096,
            padding_side="right",
            use_fast=False,
        )
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "hermes-2"
        ]

    @requires_lustre()
    def test_preprocess_mpt_image(self):
       processed_prompt = dataset.preprocess_mpt(TEST_SOURCE_IMAGE, self.tokenizer, has_image=True, no_system_prompt=False)
       assert torch.sum(processed_prompt['labels']) != 0

    @requires_lustre()
    def test_preprocess_mpt_text_only(self):
       processed_prompt = dataset.preprocess_mpt(TEST_SOURCE_TEXT_ONLY, self.tokenizer, has_image=False, no_system_prompt=False)
       assert torch.sum(processed_prompt['labels']) != 0


if __name__ == "__main__":
    unittest.main()