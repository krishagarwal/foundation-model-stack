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

    def forward(self, input: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        # return F.linear(input, self.weight, self.bias)
        input, input_scale, input_offset = input
        assert input.dtype == self.qdtype, f"input: {input.dtype}, self: {self.qdtype}"
        a, a_s, a_o = input, input_scale.type(self.dtype), input_offset.type(self.dtype)
        b, b_s, b_o = self.weight, self.weight_scale, self.weight_offset
        k = self.in_features

        # a * b = c
        # (a * as + ao) * (b * bs + bo) = (ab)(as*bs) + (a)(as*bo) + (b)(bs*ao) + I(ao*bo)

        ab_dq = (a.type(self.accdtype) @ b.type(self.accdtype)).type(self.dtype) * a_s * b_s
        a_dq = (a.type(self.dtype) * a_s) @ b_o.expand(k, 1)
        b_dq = a_o.expand(1, k) @ (b.type(self.dtype) * b_s)
        # since it's actually a k-sized dot product of a_o * b_o
        abo = k * a_o * b_o

        ab = ab_dq + a_dq + b_dq + abo

        return ab

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def custom_load(self, weight, key_steps, get_pre_rot, get_post_rot):
        weight = weight.type(self.dtype).T # transpose weights, since the linear layer is y = xA^T
        pre_rot = get_pre_rot(key_steps)
        post_rot = get_post_rot(key_steps)

        if pre_rot is not None or post_rot is not None:
            weight = weight.cuda() # TODO: remove
            if pre_rot is not None:
                weight = pre_rot @ weight
            if post_rot is not None:
                weight = weight @ post_rot
            weight = weight.cpu() # TODO: remove

        # can just cast for basic fp types
        if self.qdtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            self.weight = torch.nn.Parameter(weight.type(self.qdtype).cuda(), requires_grad=False) # TODO: remove
            self.weight_scale = torch.tensor(1, dtype=self.dtype).to(self.weight.device)
            self.weight_offset = torch.tensor(0, dtype=self.dtype).to(self.weight.device)
            return
        
        temp_weight, self.weight_scale, self.weight_offset = utils.quantize(weight, self.qdtype, torch.device('cuda'))
        self.weight = torch.nn.Parameter(temp_weight, requires_grad=False) # TODO: remove
