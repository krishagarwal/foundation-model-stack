import torch
from torch import Tensor
from torch.nn import init
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import math
from . import utils


class Linear(Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int,
                dtype, qdtype, accdtype, device=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qdtype = qdtype
        self.dtype = dtype
        self.accdtype = accdtype
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # .type(self.dtype) for scale and offset


        # if bias:
        #     self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        # else:
        #     self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     init.uniform_(self.bias, -bound, bound)

    def part_mmul(self, a: Tensor, b: Tensor):
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
        res = torch.zeros((a.shape[0], a.shape[1], b.shape[-1]), dtype=self.accdtype, device=a.device)
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
    def torch_part_mmul(self, a: Tensor, b: Tensor):
        return a.type(self.accdtype) @ b.type(self.accdtype)

    def forward(self, input: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        # return F.linear(input, self.weight, self.bias)
        input, input_scale, input_offset = input
        assert input.dtype == self.qdtype, f"input: {input.dtype}, self: {self.qdtype}"
        a, a_s, a_o = input, input_scale.type(self.dtype), input_offset.type(self.dtype)
        b, b_s, b_o = self.weight, self.weight_scale, self.weight_offset
        k = self.in_features

        # a * b = c
        # (a * as + ao) * (b * bs + bo) = (ab)(as*bs) + (a)(as*bo) + (b)(bs*ao) + I(ao*bo)
        mul_func = self.part_mmul if self.qdtype in [torch.int8, torch.int16, torch.int32] else self.torch_part_mmul
        # if self.qdtype in [torch.int8, torch.int16, torch.int32]:
        #     aab_dq = (a.type(self.accdtype) @ b.type(self.accdtype)).type(self.dtype) * a_s * b_s
        #     a_dq = (a.type(self.dtype) * a_s) @ b_o.expand(k, 1)
        #     b_dq = a_o.expand(1, k) @ (b.type(self.dtype) * b_s)
        # else:
            # ab_dq = (a.type(self.accdtype) @ b.type(self.accdtype)).type(self.dtype) * a_s * b_s
            # a_dq = (a.type(self.dtype) * a_s) @ b_o.expand(k, 1)
            # b_dq = a_o.expand(1, k) @ (b.type(self.dtype) * b_s)
        # since it's actually a k-sized dot product of a_o * b_o

        ab_dq = (mul_func(a, b).type(torch.float32) * a_s * b_s).type(self.dtype) # TODO: for fp8 quantization, do the actual mul in fp8, accumilate fp32 # TODO: don't hardcode float32. It is needed cause int32 accum > fp16 range
        a_dq = (a.type(self.dtype) * a_s) @ b_o.expand(k, 1)
        b_dq = a_o.expand(1, k) @ (b.type(self.dtype) * b_s)

        abo = k * a_o * b_o

        ab = ab_dq + a_dq + b_dq + abo

        return ab

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def custom_load(self, weight, key_steps, get_pre_rot, get_post_rot):
        try:
            weight = weight.type(self.dtype).T # transpose weights, since the linear layer is y = xA^T
            pre_rot = get_pre_rot(key_steps)
            post_rot = get_post_rot(key_steps)

            if pre_rot is not None or post_rot is not None:
                if pre_rot is not None:
                    weight = pre_rot @ weight
                if post_rot is not None:
                    weight = weight @ post_rot

            # can just cast for basic fp types
            if self.qdtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                self.weight = torch.nn.Parameter(weight.type(self.qdtype), requires_grad=False) # .to(torch.get_default_device())
                self.weight_scale = torch.tensor(1, dtype=self.dtype).to(self.weight.device)
                self.weight_offset = torch.tensor(0, dtype=self.dtype).to(self.weight.device)
                return
            
            temp_weight, self.weight_scale, self.weight_offset = utils.quantize(weight, self.qdtype) # , torch.get_default_device())
            self.weight = torch.nn.Parameter(temp_weight, requires_grad=False)
        except Exception as e:
            raise Exception(repr(e))
