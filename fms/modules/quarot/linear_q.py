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
        self.is_quantized = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def int_mmul(self, a: Tensor, b: Tensor):
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

    def fp8_mmul(self, a: Tensor, b: Tensor):
        # return torch.cat([torch._scaled_mm(a[i], b, out_dtype=utils.accdtype) for i in range(a.shape[0])]) # TODO: .T or rearrange b to be column major?
        # return a.type(self.accdtype) @ b.type(self.accdtype)

        # TODO: handle any dimensions like above
        assert (len(a.shape) == 3)
        res = torch.zeros((a.shape[0], a.shape[1], b.shape[-1]), dtype=self.accdtype, device=a.device)
        # if a.shape[1] < 16: # TODO: check if more efficient way. int8 mmul seems to require 16 min dimension, maybe cause 16x16x16 tensor core
        #     a_scratch = torch.zeros(a.shape[0], 16, a.shape[2])
        for idx in range(a.shape[0]):
            temp = res[idx]
            a_part = a[idx]
            # if a.shape[1] % 17:
            pad_val = (0, 0, 0, (16 - a_part.shape[0] % 16) % 16)
            a_part = torch.nn.functional.pad(a_part, pad_val)
            temp = torch.nn.functional.pad(temp, pad_val)

            # TODO: has strange restrictions, optimize (e.g. can we avoid requiring 17 leading dimension? we usually use only 1)
            # TODO: check what amax _ is
            temp, _ = torch._scaled_mm(a_part, b) # a_part, b, out=temp)

            # if a.shape[1] < 17:
            temp = temp[:a.shape[1]]

            res[idx] = temp
        return res

    def basic_mmul(self, a: Tensor, b: Tensor):
        return a.type(utils.accdtype) @ b.type(utils.accdtype)

    def forward(self, input) -> Tensor:
        # return F.linear(input, self.weight, self.bias)
        if not self.is_quantized: # input will be Tensor
            return self.basic_mmul(input, self.weight)
        
        # input is tuple if quantized
        input, input_scale = input
        assert input.dtype == self.qdtype, f"input: {input.dtype}, self: {self.qdtype}"
        
        a, a_s = input, input_scale.type(self.dtype)
        b, b_s = self.weight, self.weight_scale

        mul_func = self.int_mmul if self.qdtype in [torch.int8, torch.int16, torch.int32] else self.fp8_mmul if self.qdtype in [torch.float8_e4m3fn] else self.basic_mmul

        # a * b = c
        # (a * as) * (b * bs) = (ab)(as*bs)
        ab = (mul_func(a, b).type(torch.float32) * (a_s * b_s)).type(utils.dtype)

        return ab

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def custom_load(self, weight, key_steps: list[str], get_pre_rot, get_post_rot, scale= None):
        try:
            weight = weight.type(self.dtype).T # transpose weights, since the linear layer is y = xA^T

            self.is_quantized = not (utils.skip_bad_layers and 'w2' in key_steps and utils.weight_check(key_steps, ['21'])) # 2, 4 9? 10? 12? 20? 21! 22nan

            pre_rot = get_pre_rot(key_steps) if self.is_quantized else None # TODO: if other special no quantize cases, modify to work for those
            post_rot = get_post_rot(key_steps)

            if pre_rot is not None or post_rot is not None:
                if pre_rot is not None:
                    weight = pre_rot @ weight
                if post_rot is not None:
                    weight = weight @ post_rot
            
            if self.is_quantized:
                # using gpt8 weights
                if scale is not None:
                    self.weight_scale = scale.T.to(weight.device)
                elif utils.use_quant_map:
                    # weight = torch.load("map_quant_weights/" + ".".join(key_steps))
                    # weight = weight.cuda()
                    weight = utils.dequantize_cluster(*utils.quantize_cluster(weight)) # TODO: dequant should be done in shared memory live
                    self.weight_scale = torch.tensor(1, dtype=self.dtype).to(weight.device) # TODO: make optional
                    torch.save(weight, "map_quant_weights_8/" + ".".join(key_steps))
                    print("map quantized", key_steps)
                # can just cast for basic fp types
                elif self.qdtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                    weight = weight.type(self.qdtype)
                    self.weight_scale = torch.tensor(1, dtype=self.dtype).to(weight.device)
                else:
                    weight, self.weight_scale = utils.quantize(weight, self.qdtype, dim=-2)
            
            temp = torch.zeros((weight.shape[1], weight.shape[0]), dtype=weight.dtype, device=weight.device) # TODO: check if weight always 2D
            temp = temp.T
            weight = temp.copy_(weight)
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
        except Exception as e:
            raise Exception(repr(e))
