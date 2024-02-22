from llava import conversation as conversation_lib
from llava.train.token_config import (
    DEFAULT_IMAGE_PATCH_TOKEN,
)
from llava.train import arguments
from llava.train import datasets_mixture
from llava.train.dataset import make_supervised_data_module
from transformers.models.siglip import (
    SiglipImageProcessor,
)
import transformers
import torch

def test_make_supervised_data_module():
    datasets_mixture.register_datasets_mixtures()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        '/lustre/fsw/portfolios/nvr/users/jasonlu/models/vicuna-1.5/vicuna-7b-v1.5',
        cache_dir='',
        model_max_length=8192,
        padding_side="right",
        use_fast=False,
        legacy=False,
    )
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    image_processor = SiglipImageProcessor.from_pretrained(
        'google/siglip-so400m-patch14-384'
    )

    data_args = arguments.DataArguments(
        datasets_mixture_name='msrvttqa', # sharegpt4v_gpt4_100k_
        is_multimodal=True,
        lazy_preprocess=True,
    )
    data_args.image_processor=image_processor
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        "vicuna_v1_1"
    ]

    data_module, extra_info = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        patch_size=14,
        image_size=384,
        n_extra_patch=0,
    )

    dataset = data_module['train_dataset']
    # dloader = torch.utils.data.DataLoader(
    #     dataset,
    #     shuffle=True,
    #     sampler=None,
    #     batch_size=1,
    #     # collate_fn=SimpleCoyoDataset.custom_collate,
    #     num_workers=2,
    # )

    # for batch in dloader:
    #     print(batch['image'].shape, batch['input_ids'].shape)
    index = 0
    dataset_len = len(data_module['train_dataset'])
    for batch in data_module['train_dataset']:
        if index % 100 == 0:
            print(f"index: {index}/{dataset_len}")
        if batch['input_ids'].shape[0] > 4096:
            print(batch['input_ids'].shape[0])
        index += 1
            
        # print(batch['image'].shape, batch['input_ids'].shape)

    # print(data_module)

test_make_supervised_data_module()