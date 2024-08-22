import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Union

import cv2
import numpy as np
import PIL
import PIL.Image

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN
from llava.media import Image, Video
from llava.train.args import DataArguments, TrainingArguments
from llava.utils.logging import logger

__all__ = ["extract_media"]


Config = Union[DataArguments, TrainingArguments]


def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        image = PIL.Image.open(image.path)
    return image


def _load_video(video_path: str, *, num_frames: int) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video_path)

    # Find the last frame as frame count might not be accurate
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        return []

    # Extract frames uniformly
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    frames = []
    for index in indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            return []
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = PIL.Image.fromarray(frame)
        frames.append(frame)
    return frames


def _extract_video(video: Video, config: Config) -> List[PIL.Image.Image]:
    num_frames = config.num_video_frames
    if getattr(config, "fps") != 0:
        logger.warning("Extracting frames from video with specified FPS is not supported yet. Ignored.")

    frames = _load_video(video.path, num_frames=num_frames)
    if not frames:
        raise ValueError(f"Video `{video.path}` has no frames")
    return frames


def extract_media(messages: List[Dict[str, Any]], config: Config) -> Dict[str, List[Any]]:
    # TODO(zhijianl): This logic will be moved to model forward function
    if config.mm_use_im_start_end:
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    else:
        image_token = DEFAULT_IMAGE_TOKEN

    media = defaultdict(list)
    for message in messages:
        prompt = message["value"]
        if isinstance(prompt, str):
            prompt = [prompt]
        text = ""
        for part in prompt:
            if isinstance(part, str):
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                image = _extract_image(part)
                text += image_token + "\n"
                media["image"].append(image)
            elif isinstance(part, Video):
                video = _extract_video(part, config)
                text += (image_token + "\n") * len(video)
                media["image"].extend(video)
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text
    return media
