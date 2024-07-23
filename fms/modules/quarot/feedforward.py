import torch
import torch.nn as nn
from . import linear_q
from . import utils
from . import fast_had_trans
from fast_hadamard_transform import hadamard_transform
from .special_had import get_had172

class GatedLinearUnit(nn.Module):

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor: float = 4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
    ):
        super(GatedLinearUnit, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        self.w1 =  linear_q.Linear(emb_dim, self.hidden_dim, dtype=utils.dtype, qdtype=utils.qdtype, accdtype=utils.accdtype)
        self.wg =  linear_q.Linear(emb_dim, self.hidden_dim, dtype=utils.dtype, qdtype=utils.qdtype, accdtype=utils.accdtype)
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = linear_q.Linear(self.hidden_dim, emb_dim, dtype=utils.dtype, qdtype=utils.qdtype, accdtype=utils.accdtype)
        self.use_bias = use_bias
        self.width = emb_dim
        self.grow_factor = hidden_grow_factor

    def reset_parameters(self):
        for layer in ["w1", "w2", "wg"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=0.02,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def forward(self, x):
        x = utils.quantize(x, utils.qdtype, clip_ratio=utils.activ_clip_ratio)
        out = self.a(self.wg(x)) * self.w1(x)
        if self.p_dropout:
            out = self.d(out)
        if not self.w2.is_no_quant_layer:
            if utils.use_hadamard: # TODO: this is fix to make sure no rotations happen when we skip quantizing a down proj layer
                # out = out.T
                # outv = out.reshape(-1, 64, 172)
                # outv = outv @ get_had172('cuda', dtype=torch.float16)
                # outv = hadamard_transform(outv.to(torch.float32).T, scale=1.0/torch.tensor(11008, dtype=torch.float32).sqrt()).T
                # # fast_had_trans.left_had(out, had_size=64)
                # out = outv.reshape(out.shape).T
                # out, old = torch.empty(out.shape, dtype=out.dtype, device=out.device), out
                # out.copy_(old)

                # TODO: cleanup

                a = out #a.T
                av = a.view(-1, 172, 64)
                av = hadamard_transform(av.to(torch.float32), scale=1.0/torch.tensor(11008, dtype=torch.float32).sqrt())#right_had(av, had_size=64)
                av = get_had172('cuda', dtype=torch.float64) @ av.to(torch.float64)
                out = av.view(a.shape)#.T
                

                # out = out @ utils.rots[3][0]
                # out = (out.reshape(-1, 172, 64).transpose(-1, -2) @ utils.had172).transpose(-1, -2).reshape_as(out)
                # out = fast_had_trans.right_had(out, had_size=64, use_graph=(out.shape[-2] == 1)) # TODO: don't hardcode
            out = utils.quantize(out, utils.qdtype, clip_ratio=utils.activ_clip_ratio)
        result = self.w2(out)
        # # TODO: remove
        # if utils.temp_layer < 32:
        #     print(f"layer {utils.temp_layer:02d} min: {result.min(): 02.4f}, max: {result.max(): 02.4f}, mean mag: {result.abs().mean(): 02.4f}, mean sq mag sqrt (max row): {result.square().mean(dim=1).sqrt().max(): 02.4f}")
        #     utils.temp_layer += 1
        return result

