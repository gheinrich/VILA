import os
import pathlib
import re
from dataclasses import dataclass

from accelerate.hooks import add_hook_to_module
from transformers import PretrainedConfig, PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


def rprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
    else:
        return print(*args, **kwargs)


def mprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if rank == 0:
            return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
        else:
            return
    else:
        return print(*args, **kwargs)


def is_local(model_name_or_path: str) -> bool:
    return os.path.isdir(model_name_or_path)


def get_checkpoint_path(output_dir: str, checkpoint_prefix: str = "checkpoint") -> str | None:
    pathlib_dir = pathlib.Path(output_dir)

    if list(pathlib_dir.glob("config.json")):
        return output_dir
    else:
        try:
            ordering_and_checkpoint_path = []
            glob_checkpoints = [
                str(x) for x in pathlib.Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)
            ]
            for path in glob_checkpoints:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
            checkpoints_sorted = sorted(ordering_and_checkpoint_path)
            return checkpoints_sorted[-1][1]
        except:
            return None


def prepare_vision_tower_config(config: PretrainedConfig, model_args: dataclass) -> None:
    config.mm_vision_select_layer = model_args.mm_vision_select_layer
    config.mm_vision_select_feature = model_args.mm_vision_select_feature

    if getattr(config, "vision_tower_config", None) is None and model_args.vision_tower:
        ## set vision configurations
        config.vision_tower = model_args.vision_tower
        config.vision_resolution = model_args.vision_resolution
        config.interpolate_mode = model_args.interpolate_mode
        ## set vision projector configurations
        config.mm_projector_type = model_args.mm_projector_type


def vision_resolution_elevation(model: PreTrainedModel, config: PretrainedConfig):
    vision_tower = model.get_vision_tower()
    if vision_tower is not None and "radio" not in config.vision_tower.lower():
        vision_tower._maybe_resize_pos_embeds(
            model=vision_tower.vision_tower,
            image_processor=vision_tower.image_processor,
            resolution=getattr(config, "vision_resolution", -1),
            interpolate_mode=getattr(config, "interpolate_mode", "linear"),
        )


def unit_test_rope_scaling(model: PreTrainedModel, config: PretrainedConfig, training_args: dataclass):
    return False
