import os
import pathlib


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

def prepare_vision_config(config, model_args):
    if getattr(config, "vision_config", None) and model_args.vision_tower:
        ## set vision configurations
        config.vision_tower = model_args.vision_tower
        config.vision_select_layer = model_args.vision_select_layer
        config.vision_select_feature = model_args.vision_select_feature
        
        try:
            config.vision_projector = model_args.vision_projector
        except:
            raise ValueError("vision_projector is not defined in model_args")