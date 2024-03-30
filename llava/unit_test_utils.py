from unittest.case import _id as __id, skip as __skip

def requires_gpu(reason=None):
    import torch
    reason = "no GPUs detected. Only test in GPU environemnts" if reason is None else reason
    if not torch.cuda.is_available():
        return __skip(reason)
    return __id


def requires_lustre(reason=None):
    import os.path as osp
    if not osp.isdir("/lustre"):
        reason = "lustre path is not avaliable." if reason is None else reason
        return __skip(reason)
    return __id


def test_make_supervised_data_module(dataset_name, max_samples=-1, batch_size=32, num_workers=16, skip_before=0):
    import torch
    import transformers
    from transformers.models.siglip import SiglipImageProcessor
    
    from llava import conversation as conversation_lib
    from llava.data.dataset import make_supervised_data_module
    from llava.train.args import DataArguments, TrainingArguments
    
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
    dataset_len = len(data_module["train_dataset"])
    from torch.utils.data import DataLoader

    dloader = DataLoader(dataset, collate_fn=data_module["data_collator"], batch_size=batch_size, num_workers=num_workers)
    dloader_len = len(dloader)
    for idx, batch in enumerate(dloader):
        if idx < skip_before:
            continue
        
        if max_samples > 0 and idx > min(max_samples, dloader_len):
            break

        info = []
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                info.append((k, v.shape))
            else:
                info.append((k, type(v)))
        print(f"[{idx}/{len(dloader)}]", info)
