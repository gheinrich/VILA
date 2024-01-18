from unittest import mock
from llava.train.transformer_normalize_monkey_patch import patched_normalize
from llava.train.train import train

if __name__ == "__main__":
    # with mock.patch('transformers.models.clip.image_processing_clip.normalize', new=patched_normalize):
    #     train()
    train()
