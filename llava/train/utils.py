import os
import pathlib
from dataclasses import dataclass
from transformers import PretrainedConfig, PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from accelerate.hooks import add_hook_to_module


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


def get_checkpoint_path(output_dir: str) -> str | None:
    pathlib_dir = pathlib.Path(output_dir)

    if list(pathlib_dir.glob("config.json")):
        return output_dir
    else:
        try:
            checkpoint_dirs = pathlib_dir.glob("checkpoint-*")
            return str(max(checkpoint_dirs))
        except:
            return None


def prepare_vision_tower_config(
    config: PretrainedConfig, model_args: dataclass
) -> None:
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
