
import torch
import torch.nn as nn

class RawVisionEncoder(nn.Module):
    def __init__(self, resolution=448, patch_size=14) -> None:
        super().__init__()
        self.resolution = resolution
        self.patch_size = patch_size
        assert resolution % patch_size == 0
        
        self.dummy = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-1] == x.shape[-2] == self.resolution
        
        n, c, h, w = x.shape
        
        x = x.view(n, c, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size)
        # to NLD
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(n, -1, c * self.patch_size * self.patch_size)
        return x + self.dummy