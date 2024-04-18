from PIL import Image
from io import BytesIO
import base64
import numpy as np
import os

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX

import tempfile
from io import BytesIO


def get_frame_from_vcap(vidcap, num_frames=10, max_fps=0.0):
    # Always return fixed number of frames.
    import cv2

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or frame_count == 0:
        print("Video file not found. return empty images.")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames, 0
    
    duration = frame_count / fps
    frame_interval = frame_count // num_frames
    if frame_interval == 0 and frame_count <= 1:
        print("frame_interval is equal to 0. return empty image.")
        return [
            Image.new("RGB", (720, 720)),
        ] * num_frames, 0
    # print("duration:", duration, "frames:", frame_count, "intervals:", frame_interval)

    images = []
    count = 0
    success = True
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    while success:
        # print("frame_count:", frame_count, "count:", count, "num_frames:", num_frames, "frame_interval:", frame_interval)
        if frame_count >= num_frames:
            success, frame = vidcap.read()
            if count in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                if len(images) >= num_frames:
                    return images, num_frames
            count += 1
        else:
            # Left padding frames if the video is not long enough
            success, frame = vidcap.read()
            if success:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                count += 1
            elif count >= 1:
                width, height = images[-1].size
                images = [Image.new("RGB", (width, height))] * (num_frames - len(images)) + images
                print("padding frames:", (num_frames - len(images)))
                return images, num_frames
            else: 
                break
    print("Did not find enough frames in the video. return empty image.")
          
    return [
        Image.new("RGB", (720, 720)),
    ] * num_frames, 0


def get_frame_from_vcap_with_fps(vidcap, num_frames=10, max_fps=0.0):
    # return fixed number of frames if the video is long enough.
    # but if the video is not too long, we return frames based on fps.
    import cv2
    import random

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or frame_count == 0:
        print("Video file not found. return empty images.")
        empty_video_frames = int(random.uniform(2, num_frames))
        return [
            Image.new("RGB", (720, 720)),
        ] * empty_video_frames, 0
    
    duration = frame_count / fps
    # If the video is too long (longer than max_fps and num_frames can support),
    # we will use lower fps to sample frames.
    if duration >= num_frames/max_fps:
        frame_interval = frame_count // num_frames

        # If the video is too short, we will skip the video if there is only one frame.
        if frame_interval == 0 and frame_count <= 1:
            print("frame_interval is equal to 0. return empty image.")
            empty_video_frames = int(random.uniform(2, num_frames))
            return [
                Image.new("RGB", (720, 720)),
            ] * empty_video_frames, 0

        images = []
        count = 0
        success = True
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

        while success:
            if frame_count >= num_frames:
                success, frame = vidcap.read()
                if count in frame_indices:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                    if len(images) >= num_frames:
                        return images, num_frames
                count += 1
            else:
                # Left padding frames if the video is not long enough
                success, frame = vidcap.read()
                if success:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    images.append(im_pil)
                    count += 1
                elif count >= 1:
                    width, height = images[-1].size
                    images = [Image.new("RGB", (width, height))] * (num_frames - len(images)) + images
                    print("padding frames:", (num_frames - len(images)))
                    return images, num_frames
                else: 
                    break
    else:
        frames_required = int(duration * max_fps)
        # frame_interval = num_frames // frames_required
        frame_indices = np.linspace(0, frame_count - 1, frames_required, dtype=int)
        # frame_interval = int(max_fps)
        # if frame_interval < 1 or frame_count <= 1:
        if frames_required <= 1:
            print("frames_required is equal to 0. return empty image.")
            empty_video_frames = int(random.uniform(2, num_frames))
            return [
                Image.new("RGB", (720, 720)),
            ] * empty_video_frames, 0
        images = []
        count = 0
        looked = 0
        success = True

        while success:
            success, frame = vidcap.read()
            if success and (looked in frame_indices):
                # print("looked:", looked, "count:", count)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                images.append(im_pil)
                count += 1
            looked += 1
        if count >= frames_required:
            return images, count
    # print("Did not find enough frames in the video. return empty image.")
    empty_video_frames = int(random.uniform(2, num_frames))
    return [
        Image.new("RGB", (720, 720)),
    ] * empty_video_frames, 0

def opencv_extract_frames(vpath_or_bytesio, frames=6, fps=0.0):
    """
    Extract frames from a video using OpenCV.

    Args:
        vpath_or_bytesio (str or BytesIO): Path to the video file or BytesIO object containing the video.
        frames (int): Number of frames to extract from the video.
        fps (float): Frames per second of the video. If 0.0, the function will extract frames at equal intervals.

    Returns:
        list: List of PIL Images extracted from the video.

    Raises:
        NotImplementedError: If the type of `vpath_or_bytesio` is not supported.
    """
    import cv2

    if isinstance(vpath_or_bytesio, str):
        vidcap = cv2.VideoCapture(vpath_or_bytesio)
        if fps > 0.0:
            return get_frame_from_vcap_with_fps(vidcap, frames, fps)
        return get_frame_from_vcap(vidcap, frames, fps)
    elif isinstance(vpath_or_bytesio, (BytesIO,)):
        # assuming mp4
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
            temp_video.write(vpath_or_bytesio.read())
            temp_video_name = temp_video.name
            vidcap = cv2.VideoCapture(temp_video_name)
            if fps > 0.0:
                return get_frame_from_vcap_with_fps(vidcap, frames, fps)
            return get_frame_from_vcap(vidcap, frames, fps)
    else:
        raise NotImplementedError(type(vpath_or_bytesio))


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    """
    Expand the given PIL image to a square shape by adding padding.

    Parameters:
    - pil_img: The PIL image to be expanded.
    - background_color: The color of the padding to be added.

    Returns:
    - The expanded PIL image.

    If the image is already square, it is returned as is.
    If the image is wider than it is tall, padding is added to the top and bottom.
    If the image is taller than it is wide, padding is added to the left and right.
    """
    width, height = pil_img.size
    if pil_img.mode == 'L':
        background_color = background_color[0]
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def process_image(image_file, data_args, image_folder):
    processor = data_args.image_processor
    if isinstance(image_file, str):
        if image_folder is not None:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
    else:
        # image is stored in bytearray
        image = image_file
    if data_args.image_aspect_ratio == "resize":
        if hasattr(data_args.image_processor, "crop_size"):
            # CLIP vision tower
            crop_size = data_args.image_processor.crop_size
        else:
            # SIGLIP vision tower
            assert hasattr(data_args.image_processor, "size")
            crop_size = data_args.image_processor.size
        image = image.resize((crop_size["height"], crop_size["width"]))
    if data_args.image_aspect_ratio == "pad":

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        # Using default behavior of the vision encoder
        # For CLIP, default is central crop
        # For Radio, default is central crop
        # For Siglip, default is resize
        # For InternVIT, default is resize
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image

def process_images(images, image_processor, model_cfg):

    model_cfg.image_processor = image_processor
    new_images = [process_image(image, model_cfg, None) for image in images]

    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def is_gemma_tokenizer(tokenizer):
    return "gemma" in tokenizer.__class__.__name__.lower()


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0] :] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
