from logging import getLogger
import math
from typing import Union, Tuple
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import _SpectralNorm

from timm.models.vision_transformer import Attention, Mlp

_EPS = 1e-5


class _SNReweight(_SpectralNorm):
    def __init__(self, weight: torch.Tensor, *args, init_norm_to_current: bool = False, alpha: float = 0.05, version: int = 2, **kwargs):
        super().__init__(weight, *args, **kwargs)

        self.alpha = alpha
        self.version = version
        self.register_buffer('_sn_version', torch.tensor(version))

        if init_norm_to_current:
            # This will set the numerator to match the denominator, which should preserve the original values
            init_scale = self._get_sigma(weight).item()
        else:
            init_scale = 1.0

        if version == 1:
            init_value = init_scale
        elif version == 2:
            t = init_scale - alpha
            if t < _EPS:
                getLogger("spectral_reparam").warn(f'The initialized spectral norm {init_scale} is too small to be represented. Setting to {_EPS} instead.')
                t = _EPS

            init_value = math.log(math.exp(t) - 1)
        else:
            raise ValueError(f'Unsupported version: {version}')

        # Make 2D so that weight decay gets applied
        self.scale = nn.Parameter(torch.tensor([[init_value]], dtype=torch.float32, device=weight.device))

    # Re-implementing this because we need to make division by sigma safe
    def _get_sigma(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            sigma = weight.norm()
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.dot(u, torch.mv(weight_mat, v))

        return sigma + self.eps

    def forward(self, weight, *args, **kwargs):
        sigma = self._get_sigma(weight, *args, **kwargs)

        if self.version == 1:
            scale = self.scale
        elif self.version == 2:
            scale = F.softplus(self.scale) + self.alpha
        else:
            raise ValueError(f'Unsupported version: {self.version}')

        scale = scale.float() / sigma.float()

        y = weight * scale
        return y

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version_key = f'{prefix}_sn_version'
        if version_key not in state_dict:
            self.version = 1
            state_dict[version_key] = torch.tensor(1)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class _AttnSNReweight(nn.Module):
    def __init__(self, weight: torch.Tensor, *args, init_norm_to_current: bool = False, renorm_values: bool = False, **kwargs):
        super().__init__()

        parts = weight.split(weight.shape[0] // 3, dim=0)

        ct = 2 if not renorm_values else 3

        self.parts = nn.ModuleList([
            _SNReweight(p, *args, init_norm_to_current=init_norm_to_current, **kwargs) if i < ct else nn.Identity()
            for i, p in enumerate(parts)
        ])

    def forward(self, weight: torch.Tensor, *args, **kwargs):
        parts = weight.split(weight.shape[0] // 3, dim=0)

        parts = [
            fn(p)
            for fn, p in zip(self.parts, parts)
        ]

        return torch.cat(parts, dim=0)


def enable_spectral_reparam(model: nn.Module,
                            n_power_iterations: int = 1,
                            eps: float = 1e-6,
                            init_norm_to_current: bool = False,
                            renorm_values: bool = True,
                            renorm_mlp: bool = True):
    print('Enabling spectral reparametrization')
    for mod in model.modules():
        if isinstance(mod, Attention):
            parametrize.register_parametrization(
                mod.qkv,
                'weight',
                _AttnSNReweight(mod.qkv.weight, n_power_iterations, dim=0, eps=eps, init_norm_to_current=init_norm_to_current, renorm_values=renorm_values),
            )
            pass
        elif isinstance(mod, Mlp) and renorm_mlp:
            parametrize.register_parametrization(
                mod.fc1,
                'weight',
                _SNReweight(mod.fc1.weight, n_power_iterations, dim=0, eps=eps, init_norm_to_current=init_norm_to_current),
            )
            parametrize.register_parametrization(
                mod.fc2,
                'weight',
                _SNReweight(mod.fc2.weight, n_power_iterations, dim=0, eps=eps, init_norm_to_current=init_norm_to_current),
            )
            pass


def configure_spectral_reparam_from_args(model: nn.Module, args):
    spectral_reparam = getattr(args, 'spectral_reparam', False)
    if isinstance(spectral_reparam, bool) and spectral_reparam:
        enable_spectral_reparam(model, init_norm_to_current=args.pretrained)
    elif isinstance(spectral_reparam, dict):
        enable_spectral_reparam(
            model,
            n_power_iterations=spectral_reparam.get('n_power_iterations', 1),
            eps=spectral_reparam.get('eps', 1e-12),
            init_norm_to_current=args.pretrained,
        )
