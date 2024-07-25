import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

# get dtype for storage and number of bits (can be different for fake quant)
def quant_dtype_to_torch_dtype(quant_dtype) -> tuple[torch.dtype, int]:
    if quant_dtype == "int8":
        return torch.int8, 8
    elif quant_dtype == "int4-fake":
        return torch.int8, 4 # fake quant since int8 not supported at the moment TODO: we'll add packing later
    else:
        raise ValueError("Unsupported quant_dtype for quantization of weights")


def quantize(weight: torch.Tensor, qdtype, bits, dim=-1, device=None, sym=True, clip_ratio=1) -> tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = weight.device
    
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

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, quant_dtype: torch.dtype, bits: int, bias: bool = True, device=None) -> None:
        self.quant_dtype = quant_dtype
        self.bits = bits
        
        # with torch.no_grad():
        #     super().__init__(in_features, out_features, bias, device, dtype)

        # can't call super() because it has required_grad
        factory_kwargs = {'device': device, 'dtype': self.quant_dtype}
        super(nn.Linear, self).__init__() # have to call grandparent now
        self.in_features = in_features
        self.out_features = out_features
        # TODO: revert to empty, this is just for debugging
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        # if bias:
        #     self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()

        if bias:
            raise ValueError("Bias is not yet supported with quantization")
        # TODO: revert to empty, this is just for debugging
        # TODO: choose the scale dtype
        self.weight_scale = torch.nn.Parameter(torch.empty((out_features, 1), device=device, dtype=torch.float16), requires_grad=False)

        # TODO: support more dtypes and assign correct function (based on self.quant_dtype)
        self.mmul_func = int_mmul
        # self.int_mmul if self.qdtype in [torch.int8, torch.int16, torch.int32] else self.fp8_mmul_triton if self.qdtype in [torch.float8_e4m3fn] else self.basic_mmul

    def forward(self, input) -> torch.Tensor:
        # TODO: check efficiency, had to change order so scale didn't reach inf
        # TODO: check if need cast to fp32 after matmul
        a, a_s = quantize(input, self.quant_dtype, self.bits) # TODO: maybe read from weight?
        return (self.mmul_func(self, a, self.weight.T).to(torch.float32) * a_s * self.weight_scale.T).to(input.dtype)
    
    def reset_parameters(self) -> None:
        self.weight.zero_()