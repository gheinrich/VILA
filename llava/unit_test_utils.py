from unittest.case import _id as __id, skip as __skip

def requires_gpu():
    import torch
    reason = "no GPUs detected. Only test in GPU environemnts"
    if not torch.cuda.is_available():
        return __skip(reason)
    return __id


def requires_lustre():
    import os.path as osp
    if not osp.isdir("/lustre"):
        reason = "lustre path is not avaliable."
        return __skip(reason)
    return __id