import torch
from fast_hadamard_transform import hadamard_transform
from . import quantized
import functools
from utils.special_had import get_hadK

def full_normed_right_hadamard(a: torch.Tensor, power_of_two_size, hadK):
    orig_shape = input.shape
    if hadK is None:
        a = a.view(-1, power_of_two_size)
        a = hadamard_transform(a, scale=torch.tensor(power_of_two_size, dtype=a.dtype, device=a.device).rsqrt())
    else:
        a = a.view(-1, hadK.shape[0], power_of_two_size)
        a = hadamard_transform(a, scale=torch.tensor(power_of_two_size, dtype=a.dtype, device=a.device).rsqrt())
        a = torch.matmul(hadK, a)
    a = a.view(orig_shape)
    return a

def partial_normed_right_hadamard(a: torch.Tensor, completed_size, remaining_size):
    # TODO: handle remaining size not being power of 2
    orig_shape = a.shape
    a = a.view(-1, remaining_size, completed_size).transpose(-1, -2)
    a = hadamard_transform(a, scale=torch.tensor(remaining_size, dtype=a.dtype, device=a.device).rsqrt())
    a = a.transpose(-1, -2).view(orig_shape)
    return a

class Linear(quantized.Linear):
    def __init__(self, in_features: int, out_features: int, dtype, full_had: bool, had_size, completed_size: int = 0, bias: bool = True, device=None) -> None:
        super().__init__(in_features, out_features, dtype, bias, device)
        if full_had:
            pow2size, hadK = get_hadK(had_size)
            self.rotate = functools.partial(full_normed_right_hadamard, power_of_two_size=pow2size, hadK=hadK)
        else:
            self.rotate = functools.partial(partial_normed_right_hadamard, completed_size=completed_size, remaining_size=had_size)
    
    def forward(self, input) -> torch.Tensor:
        input = self.rotate(input)
        return super().forward(input)
