# This is a fix to make sure the non-pytorch fast_hadamard_transform continues to work with torch.compile without graph breaks

import torch
from fast_hadamard_transform import hadamard_transform

torch.library.define("hadamard::transform", "(Tensor a, float scale) -> Tensor")

@torch.library.impl("hadamard::transform", "cuda")
def had_trans_custom_op(a: torch.Tensor, scale: float):
    return hadamard_transform(a, scale)

@torch.library.impl_abstract("hadamard::transform")
def had_trans_faketensor_op(a: torch.Tensor, scale: float):
    return torch.empty(a.shape, dtype=a.dtype, device=a.device)