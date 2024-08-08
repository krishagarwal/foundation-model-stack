from typing import Optional, Sequence, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

torch.ops.import_module("fms.utils.int8_mmul")

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

def pack_int4(a: torch.Tensor) -> torch.Tensor:
    a = a.view(*a.shape[:-1], a.shape[-1] // 2, 2)
    a1, a2 = torch.chunk(a, 2, -1) # TODO: consider not hardcoding -1 dim
    a = (a1 << 4) | (a2 & 0b1111)
    return a.squeeze(-1)

def unpack_int4(a: torch.Tensor) -> torch.Tensor:
    a1, a2 = a >> 4, (a << 4) >> 4 # need left + right shift here for sign extension to work
    return torch.cat((a1.unsqueeze(-1), a2.unsqueeze(-1)), -1).view(*a.shape[:-1], a.shape[-1] * 2) # TODO: consider not hardcoding -1 dim

def quantize(weight: torch.Tensor, qdtype, bits, dim=-1, sym=True, clip_ratio=1) -> Tuple[torch.Tensor, torch.Tensor]:
    assert qdtype in [torch.int8, torch.uint8], "Quantize only supports int8 or uint8"
    assert sym ^ (qdtype in [torch.uint8]), "Symmetric must use signed, assymetric must use unsigned dtype"
    
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

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, quant_dtype: torch.dtype, bits: int, clip_ratio: float, bias: bool = True, device=None) -> None:
        super().__init__()
        self.quant_dtype = quant_dtype
        self.bits = bits
        factory_kwargs = {'device': device, 'dtype': self.quant_dtype}
        self.in_features = in_features
        self.out_features = out_features

        # TODO: this case is kind of hard-coded, check if there's a cleaner way
        if bits == 4:
            self.weight = Parameter(torch.empty((out_features, in_features // 2), **factory_kwargs), requires_grad=False)
        else:
            self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        if bias:
            raise ValueError("Bias is not yet supported with quantization")
        # TODO: consider choosing the scale dtype
        self.weight_scale = torch.nn.Parameter(torch.empty((out_features, 1), device=device, dtype=torch.float16), requires_grad=False)
        # TODO: support more dtypes and assign correct function (based on self.quant_dtype)
        self.mmul_func = torch.ops.quantized.int8_mmul if bits == 8 else torch.ops.quantized.int8_mmul_unpack_int4 #int_mmul
        # self.int_mmul if self.qdtype in [torch.int8, torch.int16, torch.int32] else self.fp8_mmul_triton if self.qdtype in [torch.float8_e4m3fn] else self.basic_mmul
        self.clip_ratio = clip_ratio
        # self.unpack = unpack_int4 if bits == 4 else lambda x: x # TODO: assumes that weights are 4 bit when only activations could be 4-bit (actually why do activations 4-bit at all?)

    def forward(self, input) -> torch.Tensor:
        # TODO: check efficiency, had to change order so scale didn't reach inf
        # TODO: check if need cast to fp32 after matmul
        a, a_s = quantize(input, self.quant_dtype, self.bits, dim=-1, sym=True, clip_ratio=self.clip_ratio) # TODO: maybe read from weight?
        return (self.mmul_func(a, self.weight.T).to(torch.float32) * a_s * self.weight_scale.T).to(input.dtype) # TODO: mmul_func can return fp32 maybe
    
    def reset_parameters(self) -> None:
        self.weight.zero_()

class QuantizedTensor:
    def __init__(self, value: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, last_dim_pack_factor: int = 1):
        self.value = value
        self.scale = scale
        self.offset = offset
        self.shape = list(self.value.shape)
        self.shape[-1] *= last_dim_pack_factor
    
    def size(self, dim: Optional[int] = None) -> torch.Size:
        if dim is None:
            return self.shape
        return self.shape[dim]
    
    def cat(self, other: "QuantizedTensor", dim: int = 0) -> "QuantizedTensor":
        value = torch.cat((self.value, other.value), dim)
        scale = torch.cat((self.scale, other.scale), dim)
        offset = torch.cat((self.offset, other.offset), dim)
        return QuantizedTensor(value, scale, offset)
    
    def numel(self) -> int:
        return self.value.numel()
    
    def unsqueeze(self, dim: int) -> "QuantizedTensor":
        return QuantizedTensor(self.value.unsqueeze(dim), self.scale.unsqueeze(dim), self.offset.unsqueeze(dim))
    
    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "QuantizedTensor":
        return QuantizedTensor(self.value.flatten(start_dim, end_dim), self.scale.flatten(start_dim, end_dim), self.offset.flatten(start_dim, end_dim))
    
    def expand(self, size: Sequence[int], *, implicit: bool = False) -> "QuantizedTensor":
        return QuantizedTensor(self.value.expand(size, implicit=implicit), self.scale.expand(size, implicit=implicit), self.offset.expand(size, implicit=implicit))
    
    def clone(self, *, memory_format: Optional[torch.memory_format] = None):
        return QuantizedTensor(self.value.clone(memory_format=memory_format), self.scale.clone(memory_format=memory_format), self.offset.clone(memory_format=memory_format))
    
    def detach(self):
        return QuantizedTensor(self.value.detach(), self.scale.detach(), self.offset.detach())

class KVCacheQuantizer:
    def __init__(self, quant_dtype: torch.dtype, bits: int, clip_ratio: float) -> None:
        if quant_dtype not in [torch.uint8]:
            raise ValueError(f"KV-cache quantization storage only supported for uint8")
        if bits not in [4, 8]:
            raise ValueError(f"KV-cache quantization only supported for 4 and 8 bits")
        
        self.quant_dtype = quant_dtype
        self.bits = bits
        self.clip_ratio = clip_ratio
        self.last_dim_pack_factor = 1 if bits == 8 else 2
        self.pack = pack_int4 if bits == 4 else lambda x: x
        self.unpack = unpack_int4 if bits == 4 else lambda x: x
    
    def quantize(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[QuantizedTensor, QuantizedTensor]:
        keys = quantize(keys, self.quant_dtype, self.bits, sym=False, dim=-1, clip_ratio=self.clip_ratio)
        values = quantize(values, self.quant_dtype, self.bits, sym=False, dim=-1, clip_ratio=self.clip_ratio)
        # keys = (self.pack(keys[0]), *keys[1:])
        # values = (self.pack(values[0]), *values[1:])
        return QuantizedTensor(*keys, self.last_dim_pack_factor), QuantizedTensor(*values, self.last_dim_pack_factor)
    
    def cat(self, keys: QuantizedTensor, values: QuantizedTensor, past_keys: QuantizedTensor, past_values: QuantizedTensor) -> Tuple[QuantizedTensor, QuantizedTensor]:
        # NOTE: this bad casting is pytorch's fault, without the casting torch.compile will not be able to fuse these
        # into one Triton kernelthis is not necessary for some pytorch2.4 nightly we used but the stable
        # pytorch 2.4 has compile broken

        new_keys = torch.cat((past_keys.value, keys.value.to(torch.float16)), dim=2).to(torch.uint8)
        new_values = torch.cat((past_values.value, values.value.to(torch.float16)), dim=2).to(torch.uint8)
        new_keys_off = torch.cat((past_keys.offset, keys.offset), dim=2)
        new_values_off = torch.cat((past_values.offset, values.offset), dim=2)
        new_keys_scale = torch.cat((past_keys.scale, keys.scale), dim=2)
        new_values_scale = torch.cat((past_values.scale, values.scale), dim=2)

        return QuantizedTensor(new_keys, new_keys_scale, new_keys_off, self.last_dim_pack_factor), QuantizedTensor(new_values, new_values_scale, new_values_off, self.last_dim_pack_factor)
    
    def dequantize(self, keys: QuantizedTensor, values: QuantizedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        keys_tensor = self.unpack(keys.value)
        values_tensor = self.unpack(values.value)
        return keys.scale * (keys_tensor - keys.offset), values.scale * (values_tensor - values.offset)
