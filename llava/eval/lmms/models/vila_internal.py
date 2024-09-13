from typing import List, Tuple

import accelerate
import torch
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.media import Video
from llava.utils import distributed as dist


@register_model("vila_internal")
class VILA(lmms):
    def __init__(
        self, model_path: str, conv_mode: str, num_video_frames: int = 8, max_tiles: int = 12, batch_size: int = 1
    ) -> None:
        super().__init__()
        assert batch_size == 1, "VILA only supports batch size of 1 at the moment."

        devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
        torch.cuda.set_device(devices[0])

        self.model = llava.load(model_path, devices=devices)
        self.model.config.num_video_frames = num_video_frames
        context_length = num_video_frames * 512

        self.model.config.min_tiles = 1
        self.model.config.max_tiles = max_tiles
        self.model.llm.config.min_tiles = 1
        self.model.llm.config.max_tiles = max_tiles

        if max_tiles > 12:
            context_length = max(context_length, int(max_tiles / 12.0 * 4096))

        self.model.config.model_max_length = context_length
        self.model.config.tokenizer_model_max_length = context_length
        self.model.llm.config.model_max_length = context_length
        self.model.llm.config.tokenizer_model_max_length = context_length

        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_mode].copy()

        self.accelerator = accelerate.Accelerator()
        self.device = torch.device(f"cuda:{devices[0]}")
        self._world_size = dist.size()
        self._rank = dist.rank()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses = []
        for request in tqdm(requests, disable=not dist.is_main()):
            prompt, generation_kwargs, doc_to_visual, doc_id, task, split = self._patch(request.args)
            doc = self.task_dict[task][split][doc_id]

            # Generate multimodal prompt
            medias = []
            for media in doc_to_visual(doc):
                if isinstance(media, str):
                    if media.endswith(".mp4"):
                        media = Video(media)
                    else:
                        raise NotImplementedError(f"Unsupported media type: {media}")
                medias.append(media)
            prompt = medias + [prompt]

            # Override generation config
            generation_config = self.model.default_generation_config
            generation_config.update(**generation_kwargs)

            # Generate response
            response = self.model.generate_content(prompt, generation_config=generation_config)
            responses.append(response)

            print("Prompt:", prompt)
            print("Response:", response)
        return responses

    def _patch(self, args: Tuple) -> Tuple:
        prompt, generation_kwargs, doc_to_visual, doc_id, task, split = args
        doc = self.task_dict[task][split][doc_id]

        if task in ["videomme", "videomme_w_subtitle"]:
            from llava.eval.lmms.tasks.videomme import (
                videomme_doc_to_text,
                videomme_doc_to_text_subtitle,
                videomme_doc_to_visual,
            )

            # NOTE(zhijianl): This is a hack to make sure the video path is correct for Video-MME.
            doc_to_visual = videomme_doc_to_visual

            # Override the prompt for Video-MME, which can offer more than 1% improvement over the default prompt.
            if task == "videomme":
                prompt = videomme_doc_to_text(doc)
            if task == "videomme_w_subtitle":
                prompt = videomme_doc_to_text_subtitle(doc, num_frames=self.model.config.num_video_frames)

        return prompt, generation_kwargs, doc_to_visual, doc_id, task, split

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
