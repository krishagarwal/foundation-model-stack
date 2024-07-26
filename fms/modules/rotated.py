from typing import Optional, Tuple
import torch
from fast_hadamard_transform import hadamard_transform
from . import quantized, positions
import functools
from fms.utils.special_had import get_hadK

torch.ops.import_module("fms.modules.hadamard")

def full_normed_right_hadamard(a: torch.Tensor, power_of_two_size, hadK):
    orig_shape = a.shape
    if hadK is None:
        a = a.view(-1, power_of_two_size)
        a = torch.ops.hadamard.transform(a, scale=torch.tensor(power_of_two_size, dtype=a.dtype, device=a.device).rsqrt())
        # a = hadamard_transform(a, scale=torch.tensor(power_of_two_size, dtype=a.dtype, device=a.device).rsqrt())
    else:
        a = a.view(-1, hadK.shape[0], power_of_two_size)
        a = torch.ops.hadamard.transform(a, scale=torch.tensor(power_of_two_size, dtype=a.dtype, device=a.device).rsqrt())
        # a = hadamard_transform(a, scale=torch.tensor(power_of_two_size, dtype=a.dtype, device=a.device).rsqrt())
        a = torch.matmul(hadK, a)
    a = a.view(orig_shape)
    return a

def partial_normed_right_hadamard(a: torch.Tensor, completed_size, remaining_size):
    # TODO: handle remaining size not being power of 2
    orig_shape = a.shape
    a = a.view(-1, remaining_size, completed_size).transpose(-1, -2)
    a = torch.ops.hadamard.transform(a, scale=torch.tensor(remaining_size, dtype=a.dtype, device=a.device).rsqrt())
    # a = hadamard_transform(a, scale=torch.tensor(remaining_size, dtype=a.dtype, device=a.device).rsqrt())
    a = a.transpose(-1, -2).reshape(orig_shape) # TODO: can we do a view?
    return a

class Linear(quantized.Linear):
    def __init__(self, in_features: int, out_features: int, quant_dtype: torch.dtype, bits: int, clip_ratio: float, is_full_had: bool, had_size: int, completed_size: Optional[int] = None, bias: bool = True, device=None) -> None:
        super().__init__(in_features, out_features, quant_dtype, bits, clip_ratio, bias, device)
        if is_full_had:
            pow2size, hadK = get_hadK(had_size)
            if hadK is not None:
                hadK = (hadK.to(device) * torch.tensor(hadK.shape[0], device=device).rsqrt()).to(torch.float16) # TODO: don't hardcode float16, consider not hardcoding float32 in special_had.py
            self.rotate = functools.partial(full_normed_right_hadamard, power_of_two_size=pow2size, hadK=hadK)
        else:
            self.rotate = functools.partial(partial_normed_right_hadamard, completed_size=completed_size, remaining_size=had_size)
        return

    def forward(self, input) -> torch.Tensor:
        input = self.rotate(input)
        return super().forward(input)

class RotaryEmbedding(positions.RotaryEmbedding):
    def __init__(self, had_size: int, dim: int, ratio: float = 10, max_seq_len=2048, ntk_scaling=False):
        super().__init__(dim, ratio, max_seq_len, ntk_scaling)
        pow2size, hadK = get_hadK(had_size)
        assert hadK is None # TODO: temporary fix since we can't cast hadK to a device without knowing the device upfront
        # TODO: consider supporting partial hadamard
        self.rotate = functools.partial(full_normed_right_hadamard, power_of_two_size=pow2size, hadK=hadK)
    
    def adjusted_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_kv_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries, keys = super().adjusted_qk(q, k, position_ids, past_kv_state, use_cache)
        qk = torch.cat([queries, keys], dim=0)
        qk = self.rotate(qk)
        queries, keys = qk.split([queries.shape[0], keys.shape[0]])
        return queries, keys
