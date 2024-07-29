from typing import Optional, Sequence, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

# get dtype for storage and number of bits (can be different for fake quant)
def quant_dtype_to_torch_dtype(quant_dtype) -> Tuple[torch.dtype, int]:
    if quant_dtype == "int8":
        return torch.int8, 8
    elif quant_dtype == "int4-fake":
        return torch.int8, 4 # fake quant since int8 not supported at the moment TODO: we'll add packing later
    else:
        raise ValueError("Unsupported quant_dtype for quantization of weights")

def signed_to_unsigned_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.int8:
        return torch.uint8
    elif dtype == torch.int16:
        return torch.uint16
    elif dtype == torch.int32:
        return torch.uint32
    elif dtype == torch.int64:
        return torch.uint64
    else:
        raise ValueError("Provided dtype does not have an associated unsigned dtype")


def quantize(weight: torch.Tensor, qdtype, bits, dim=-1, sym=True, clip_ratio=1) -> Tuple[torch.Tensor, torch.Tensor]:
    assert qdtype in [torch.int8, torch.uint8], "Quantize only supports int8 or uint8"
    assert sym ^ (qdtype in [torch.uint8]), "Symmetric must use signed, assymetric must use unsigned dtype"

    max_qint = 2 ** (bits - 1) - 1 if sym else 2 ** bits - 1
    min_qint = -(max_qint + 1) if sym else 0
    
    if sym:
        max_qint = 2 ** (bits - 1) - 1
        min_qint = -(max_qint + 1)

        mag, _ = weight.abs().max(dim=dim, keepdim=True)
        mag *= clip_ratio
        is_zero = mag == 0
        scale = torch.where(is_zero, 1, mag / max_qint)
        weight = weight / scale
        weight = weight.clamp(min=min_qint, max=max_qint).round()

        return weight.to(qdtype), scale
    
    else:
        max_qint = 2 ** bits - 1
        min_qint = 0

        # don't let min or max cross over zero
        weight_max = weight.max(dim=dim, keepdim=True)[0].maximum(torch.tensor(0, dtype=weight.dtype, device=weight.device)) * clip_ratio
        weight_min = weight.min(dim=dim, keepdim=True)[0].minimum(torch.tensor(0, dtype=weight.dtype, device=weight.device)) * clip_ratio
        
        is_zero = (weight_max == 0) & (weight_min == 0)
        weight_max[is_zero] = 1
        weight_min[is_zero] = -1
        mag = weight_max - weight_min
        scale = mag / max_qint
        offset = torch.round(-weight_min / scale)
        weight = weight / scale + offset
        weight = weight.clamp(min=min_qint, max=max_qint).round()

        return weight.to(qdtype), scale, offset

def int8_mmul(self, a: torch.Tensor, b: torch.Tensor):
        res = torch.zeros((*a.shape[:-1], b.shape[-1]), dtype=torch.int32, device=a.device)
        idxs = list((x,) for x in range(input.shape[-3]))

        # TODO: bad fix, try batching multiple things into one mmul
        for i in range(-4, -len(input.shape) - 1, -1):
            idxs = [(curr, *idx) for idx in idxs for curr in range(input.shape[i])]
            
        # assert (len(a.shape) == 3)
        # res = torch.zeros((a.shape[0], a.shape[1], b.shape[-1]), dtype=self.accdtype, device=a.device)
        # TODO: check if more efficient way. int8 mmul seems to require 16 min dimension, maybe cause 16x16x16 tensor core
        for idx in range(idxs):
            temp = res[idx]
            a_part = a[idx]
            if a.shape[1] < 17:
                pad_val = (0, 0, 0, 17 - a_part.shape[0])
                a_part = F.pad(a_part, pad_val)
                temp = F.pad(temp, pad_val)

            # TODO: has strange restrictions, optimize (e.g. can we avoid requiring 17 leading dimension? we usually use only 1)
            torch._int_mm(a_part, b.T, out=temp)

            if a.shape[1] < 17:
                temp = temp[:a.shape[1]]

            res[idx] = temp
        return res

def int_mmul(self, a: torch.Tensor, b: torch.Tensor):
    # res = torch.zeros((*a.shape[:-1], b.shape[-1]), dtype=self.accdtype, device=a.device)
    # idxs = list((x,) for x in range(input.shape[-3]))
    # for i in range(-4, -len(input.shape) - 1, -1):
    #     idxs = [(curr, *idx) for idx in idxs for curr in range(input.shape[i])]
        
    # for idx in idxs:
    #     temp = res[idx]
    #     torch._int_mm(a[idx].type(self.accdtype), b.type(self.accdtype), out=temp)
    #     res[idx] = temp
    # return res

    # TODO: handle any dimensions like above
    assert (len(a.shape) == 3)
    res = torch.zeros((a.shape[0], a.shape[1], b.shape[-1]), dtype=torch.int32, device=a.device) # TODO: don't hardcode
    # if a.shape[1] < 16: # TODO: check if more efficient way. int8 mmul seems to require 16 min dimension, maybe cause 16x16x16 tensor core
    #     a_scratch = torch.zeros(a.shape[0], 16, a.shape[2])
    for idx in range(a.shape[0]):
        temp = res[idx]
        a_part = a[idx]
        if a.shape[1] < 17:
            pad_val = (0, 0, 0, 17 - a_part.shape[0])
            a_part = torch.nn.functional.pad(a_part, pad_val)
            temp = torch.nn.functional.pad(temp, pad_val)

        # TODO: has strange restrictions, optimize (e.g. can we avoid requiring 17 leading dimension? we usually use only 1)
        torch._int_mm(a_part, b, out=temp)

        if a.shape[1] < 17:
            temp = temp[:a.shape[1]]

        res[idx] = temp
    return res

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, quant_dtype: torch.dtype, bits: int, clip_ratio: float, bias: bool = True, device=None) -> None:
        super().__init__()
        self.quant_dtype = quant_dtype
        self.bits = bits
        factory_kwargs = {'device': device, 'dtype': self.quant_dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        # if bias:
        #     self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()
        if bias:
            raise ValueError("Bias is not yet supported with quantization")
        # TODO: consider choosing the scale dtype
        self.weight_scale = torch.nn.Parameter(torch.empty((out_features, 1), device=device, dtype=torch.float16), requires_grad=False)
        # TODO: support more dtypes and assign correct function (based on self.quant_dtype)
        self.mmul_func = int_mmul
        # self.int_mmul if self.qdtype in [torch.int8, torch.int16, torch.int32] else self.fp8_mmul_triton if self.qdtype in [torch.float8_e4m3fn] else self.basic_mmul
        self.clip_ratio = clip_ratio

    def forward(self, input) -> torch.Tensor:
        # TODO: check efficiency, had to change order so scale didn't reach inf
        # TODO: check if need cast to fp32 after matmul
        a, a_s = quantize(input, self.quant_dtype, self.bits, dim=-1, sym=True, clip_ratio=self.clip_ratio) # TODO: maybe read from weight?
        return (self.mmul_func(self, a, self.weight.T).to(torch.float32) * a_s * self.weight_scale.T).to(input.dtype)
    
    def reset_parameters(self) -> None:
        self.weight.zero_()

class QuantizedTensor:
    def __init__(self, value: torch.Tensor, scale: torch.Tensor, offset: Optional[torch.Tensor] = None):
        self.value = value
        self.scale = scale
        self.offset = offset
    
    @property
    def shape(self) -> torch.Size:
        return self.value.shape
    
    def size(self, dim: None = None) -> torch.Size:
        return self.value.size(dim)
    
    def dequantize(self) -> torch.Tensor:
        if self.offset is None:
            return self.scale * self.value
        return self.scale * (self.value - self.offset)
    
    def cat(self, other: "QuantizedTensor", dim: int = 0) -> "QuantizedTensor":
        value = torch.cat((self.value, other.value), dim)
        scale = torch.cat((self.scale, other.scale), dim)
        if self.offset is not None:
            offset = torch.cat((self.offset, other.offset), dim) # TODO: maybe assert that both have/don't have offset
        else:
            offset = None
        return QuantizedTensor(value, scale, offset)
    
    def numel(self) -> int:
        return self.value.numel()
    
    def unsqueeze(self, dim: int) -> "QuantizedTensor":
        return QuantizedTensor(self.value.unsqueeze(dim), self.scale.unsqueeze(dim), self.offset.unsqueeze(dim) if self.offset is not None else None)
    
    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "QuantizedTensor":
        return QuantizedTensor(self.value.flatten(start_dim, end_dim), self.scale.flatten(start_dim, end_dim), self.offset.flatten(start_dim, end_dim) if self.offset is not None else None)
    
    def expand(self, size: Sequence[int], *, implicit: bool = False) -> "QuantizedTensor":
        return QuantizedTensor(self.value.expand(size, implicit=implicit), self.scale.expand(size, implicit=implicit), self.offset.expand(size, implicit=implicit) if self.offset is not None else None)

class KVCacheQuantizer:
    def __init__(self, quant_dtype: torch.dtype, bits: int, sym: bool, clip_ratio: float) -> None:
        self.quant_dtype = quant_dtype
        self.bits = bits
        self.sym = sym # TODO: consider not allowing symmetric
        self.clip_ratio = clip_ratio
    
    def quantize(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[QuantizedTensor, QuantizedTensor]:
        keys = quantize(keys, self.quant_dtype, self.bits, sym=self.sym, dim=-1, clip_ratio=self.clip_ratio)
        values = quantize(values, self.quant_dtype, self.bits, sym=self.sym, dim=-1, clip_ratio=self.clip_ratio)
        return QuantizedTensor(*keys), QuantizedTensor(*values)
    
    def dequantize(self, keys: QuantizedTensor, values: QuantizedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return keys.dequantize(), values.dequantize()