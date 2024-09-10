import glob
import json
import os
import warnings
from pprint import pprint
from typing import Any, Dict, List

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.data.base import BaseDataset
from llava.media import Image
from llava.utils import io
from llava.utils.logging import logger

__all__ = ["LLaVADataset", "LLaVANextDataset", "LLaVANextVideoDataset"]


class LLaVADataset(BaseDataset):
    def __init__(self, data_path: str, image_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.image_dir = image_dir
        self.instances = []
        self.enable_dynamic_res = True
        for instance in io.load(self.data_path):
            if "image" in instance:
                image_path = os.path.join(self.image_dir, instance.pop("image"))
                if not os.path.exists(image_path):
                    logger.warning(f"Image `{image_path}` not found. Excluded from dataset.")
                    continue
                instance["image_path"] = image_path
            self.instances.append(instance)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = instance["conversations"]
        if "image_path" in instance:
            # Remove the image token from the messages
            for message in instance["conversations"]:
                message["value"] = message["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()

            # Add image to the first message
            image = Image(instance["image_path"])
            messages[0]["value"] = [image, messages[0]["value"]]
        return messages


"""
{
"sample_id": 5390,
"conversations": [
    {
    "from": "human",
    "value": "<image><image>\nWhat's the detailed difference between the 2 images? Please list in detail."
    },
    {
    "from": "gpt",
    "value": "The differences between the two images are:\n\n1. In the second image, there are leaves falling from the sunflowers and the surrounding foliage.\n2. The ground in the second image is covered with a layer of fallen leaves, adding a carpet-like appearance."
    }
],
"image": [
    "HQ-Edit/images/83425.jpg",
    "HQ-Edit/images/83426.jpg"
],
"choice_list": null,
"metadata": {
    "dataset": "HQ-Edit-Diff",
    "split": "train",
    "num_sample": 98675,
    "task_instruction": "What's the difference between 2 images?",
    "question_type": "open-ended"
}
},
"""


class LLaVANextDataset(BaseDataset):
    def __init__(self, data_path: str, image_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.image_dir = image_dir
        self.instances = io.load(self.data_path)
        self.enable_dynamic_res = True

    def process(self, instance: Dict[str, Any], index: int = None) -> List[Dict[str, Any]]:
        """
        "<image> <image> text text <image>"
        =>
        [Image, Image, text, text, Image]
        """
        datasource = instance.get("datasource", None)
        messages = instance["conversations"]

        if "image" in instance:
            img_list = []
            for img_path in instance["image"]:
                img_list.append(Image(os.path.join(self.image_dir, img_path)))

            # replace all <image> tokens in the messages
            for idx1, msg in enumerate(messages):
                # value = messages[0]["value"]
                value = messages[idx1]["value"]
                img_tok_len = len(DEFAULT_IMAGE_TOKEN)
                new_value = []

                while value.find(DEFAULT_IMAGE_TOKEN) >= 0:  # still has <image>
                    idx = value.find(DEFAULT_IMAGE_TOKEN)
                    if idx > 0:
                        new_value.append(value[:idx])
                    new_value.append(img_list.pop(0))
                    value = value[idx + img_tok_len :]
                new_value.append(value)
                messages[idx1]["value"] = new_value

                # FIXME(ligeng): this is an interesting bug... if we feed [{"from": "gpt"}, {"from": "user"}] to the model, it will throw errors.
                if datasource == "twitter_post":
                    # warnings.warn(f"{index} {datasource} enforcing the role for twitter_post datasource")
                    role = "human" if idx1 % 2 == 0 else "gpt"
                    messages[idx1]["from"] = role

            assert (
                len(img_list) == 0
            ), f"#Num of <images> does not match the number of images in the instance. {instance}"
        return messages


"""
{
    "video": "v_XNTy5ZTMqVU-Scene-011",
    "conversations": [
        {
        "from": "human",
        "value": "<image>\nWhat is the setting of the video?"
        },
        {
        "from": "gpt",
        "value": "The setting of the video appears to be a newsroom with desks and computers in the background, suggesting an office environment with other personnel working."
        }
    ],
    "id": "v_XNTy5ZTMqVU-Scene-011_1",
    "sample_id": 1,
    "metadata": {
        "dataset": "Video_VQA_Captioning",
        "split": "train",
        "task_instruction": "",
        "num_sample": 255000,
        "question_type": "open-ended"
    }
}
"""


class LLaVANextVideoDataset(BaseDataset):
    def __init__(self, data_path: str, image_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.image_dir = image_dir
        self.instances = io.load(self.data_path)

    def process(self, instance: Dict[str, Any], index: int = None) -> List[Dict[str, Any]]:
        datasource = instance.get("datasource", None)
        messages = instance["conversations"]

        if "video" in instance:

            img_flist = glob.glob(os.path.join(self.image_dir, instance["video"]) + "/*.jpeg")
            vpath = os.path.join(self.image_dir, instance["video"])

            assert len(img_flist) > 0, f"no images found in {vpath}"
            value = messages[0]["value"]
            img_list = [Image(img_path) for img_path in img_flist]
            new_value = [*img_list, value.replace(DEFAULT_IMAGE_TOKEN, "").strip()]
            messages[0]["value"] = new_value
        return messages
