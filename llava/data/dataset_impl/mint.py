import os
from typing import Any, Dict, List

from PIL import Image
from wids import ShardListDataset

from llava.data.base import BaseDataset

__all__ = ["MINTArXivDataset"]


class MINTArXivDataset(BaseDataset):
    def __init__(self, data_path: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.instances = ShardListDataset(self.data_path, cache_dir=os.path.expanduser("~/.cache/wids"))

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        captions = instance[".json"]["captions"]
        if captions[0] is None:
            raise ValueError(f"No captions found in the instance {instance}")
        caption = captions[0]

        image = Image.open(instance[".tiff"])

        messages = []
        messages.append({"from": "human", "value": [image, "What is the caption for this figure?"]})
        messages.append({"from": "gpt", "value": caption})
        return messages
