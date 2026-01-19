import torch
import torch.nn.functional as F


def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    out.copy_(F.silu(x[..., :d]) * x[..., d:])
    return out


class MyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), device=x.device, dtype=x.dtype)
        return silu_and_mul(out, x)
