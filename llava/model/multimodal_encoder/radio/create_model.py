from torch import nn

from timm.models import create_model

# This will trigger timm's register_model, which allows these architectures
# to be instantiated
from .extra_timm_models import *
from .enable_cpe_support import enable_cpe


def create_model_from_args(args):
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    # args.num_classes = 21843

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **args.model_kwargs,
    )

    if hasattr(model, 'norm') and not getattr(args, 'model_norm', False):
        model.norm = nn.Identity()

    assert not args.cls_token_per_teacher or args.cpe_max_size is not None, "CPE must be enabled for multiple CLS tokens!"

    if args.cpe_max_size is not None:
        enable_cpe(model,
                   args.cpe_max_size,
                   num_cls_tokens=len(args.teachers) if args.cls_token_per_teacher else 1,
                   register_multiple=args.register_multiple,
        )



    return model
