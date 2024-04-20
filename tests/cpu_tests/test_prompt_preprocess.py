import unittest
import torch
from llava.data import dataset
import transformers
from llava.unit_test_utils import requires_lustre
from llava import conversation as conversation_lib


TEST_SOURCE_IMAGE = [[{"from": "human", "value": "<image>\nWho is the author of this book?\nAnswer the question using a single word or phrase."}, {"from": "gpt", "value": "*"}, {"from": "human", "value": "What is the title of this book?"}, {"from": "gpt", "value": "Rough Guide Map of South Africa, Lesotho"}, {"from": "human", "value": "What is the genre of this book?"}, {"from": "gpt", "value": "Travel"}, {"from": "human", "value": "Is this book related to Travel?"}, {"from": "gpt", "value": "Yes"}, {"from": "human", "value": "Is this book related to History?"}, {"from": "gpt", "value": "No"}]]
TEST_SOURCE_IMAGE2 = [[{'from': 'human', 'value': '<image>\nWhat do you see happening in this image?'}, {'from': 'gpt', 'value': "In the image, a person's hand is seen holding a pink highlighter pen, which is being used to highlight text in a notebook. The notebook is open, revealing a page filled with handwritten text. The text appears to be in a foreign language, suggesting that the person might be studying or learning a new language. The background of the image is blurred, drawing focus to the hand and the notebook. The overall scene suggests a learning or study environment."}]]
TEST_SOURCE_TEXT_ONLY = [[{"from": "human", "value": "Who is the author of this book?\nAnswer the question using a single word or phrase."}, {"from": "gpt", "value": "*"}, {"from": "human", "value": "What is the title of this book?"}, {"from": "gpt", "value": "Rough Guide Map of South Africa, Lesotho"}, {"from": "human", "value": "What is the genre of this book?"}, {"from": "gpt", "value": "Travel"}, {"from": "human", "value": "Is this book related to Travel?"}, {"from": "gpt", "value": "Yes"}, {"from": "human", "value": "Is this book related to History?"}, {"from": "gpt", "value": "No"}]]


def compare_output(tokenizer, labels, sources):
    start = 0
    end = 0
    outputs = []
    for i, label in enumerate(labels):
        if i == 0:
            continue
        if label != -100 and labels[i-1] == -100:
            start = i
        if i == len(labels) - 1 or (label != -100 and labels[i+1] == -100):
            end = i
            outputs.append(tokenizer.decode(labels[start:end+1]))

    expected_outputs = []
    for source in sources[0]:
        if source['from'] == 'gpt':
            expected_outputs.append(source['value']+tokenizer.eos_token)
    print(expected_outputs)
    print(outputs)

    for expected, output in zip(expected_outputs, outputs):
        assert expected == output.lstrip()


class TestVicunaPromptPreprocess(unittest.TestCase):

    def setUp(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            '/home/jasonlu/models/vicuna-1.5/vicuna-7b-v1.5',
            cache_dir='',
            model_max_length=4096,
            padding_side="right",
            use_fast=False,
            legacy=False
        )
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "vicuna_v1"
        ]

    @requires_lustre()
    def test_preprocess_v2_image(self):
        for sources in [TEST_SOURCE_IMAGE, TEST_SOURCE_IMAGE2, TEST_SOURCE_TEXT_ONLY]:
            processed_prompt = dataset.preprocess_v1(sources, self.tokenizer, has_image=True, no_system_prompt=False)
            labels = processed_prompt['labels'][0]
            compare_output(self.tokenizer, labels, sources)


class TestLlama3PromptPreprocess(unittest.TestCase):

    def setUp(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            '/home/ligengz/downloads/Meta-Llama-3-8B',
            cache_dir='',
            model_max_length=4096,
            padding_side="right",
            use_fast=False,
            legacy=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens(dict(pad_token="[PAD]"))
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            "llama_3"
        ]

    @requires_lustre()
    def test_preprocess_v2_image(self):
        for sources in [TEST_SOURCE_IMAGE, TEST_SOURCE_IMAGE2, TEST_SOURCE_TEXT_ONLY]:
            processed_prompt = dataset.preprocess_v2(sources, self.tokenizer, has_image=True, no_system_prompt=False)
            labels = processed_prompt['labels'][0]
            compare_output(self.tokenizer, labels, sources)


class TestYiPromptPreprocess(unittest.TestCase):

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
    def test_preprocess_v2_image(self):
        for sources in [TEST_SOURCE_IMAGE, TEST_SOURCE_IMAGE2, TEST_SOURCE_TEXT_ONLY]:
            processed_prompt = dataset.preprocess_mpt(sources, self.tokenizer, has_image=True, no_system_prompt=False)
            labels = processed_prompt['labels'][0]
            compare_output(self.tokenizer, labels, sources)


if __name__ == "__main__":
    unittest.main()