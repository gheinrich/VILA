from unittest import mock
from llava.train.transformer_normalize_monkey_patch import patched_normalize
from llava.train.train import train

def __len__(self):
    return len(self.batch_sampler)


def __iter__(self):
    return self.batch_sampler.__iter__()


if __name__ == "__main__":
    with (
        mock.patch('transformers.image_processing_utils.normalize', new=patched_normalize),
        mock.patch('accelerate.data_loader.BatchSamplerShard.__len__', new=__len__),
        mock.patch('accelerate.data_loader.BatchSamplerShard.__iter__', new=__iter__)
        ):
            train()
