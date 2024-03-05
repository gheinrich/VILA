from unittest.case import _id as __id, skip as __skip

def requires_gpu(reason=None):
    import torch
    reason = "no GPUs detected. Only test in GPU environemnts" if reason is None else reason
    if not torch.cuda.is_available():
        return __skip(reason)
    return __id


def requires_lustre(reason=None):
    import os.path as osp
    if not osp.isdir("/lustre"):
        reason = "lustre path is not avaliable." if reason is None else reason
        return __skip(reason)
    return __id


