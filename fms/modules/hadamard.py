# This is a fix to make sure the non-pytorch faster_hadamard_transform continues to work with torch.compile without graph breaks

import torch
from fast_hadamard_transform import hadamard_transform

torch.library.define("hadamard::transform", "(Tensor a) -> Tensor")

@torch.library.impl("hadamard::transform", "cuda")
def had_trans_custom_op(a: torch.Tensor):
    return hadamard_transform(a)

@torch.library.impl_abstract("hadamard::transform")
def had_trans_faketensor_op(a: torch.Tensor):
    return torch.empty(a.shape, dtype=a.dtype, device=a.device)