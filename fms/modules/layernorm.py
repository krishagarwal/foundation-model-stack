import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import dropout_layer_norm


class LayerNormParameterized(nn.Module):
    """
    A generalized LayerNorm implementation. With all optional arguments set to True, equivalent to nn.LayerNorm up to epsilon stabilization term
    (this class divides inputs by min(norm, eps), while nn.LayerNorm divides by norm + eps).
    ...
    Args
    ----
    normalized_shape : int
        Dimensionality of input data (size of final tensor axis)
    eps : float
        Safety term to prevent division by zero. Make sure the chosen value fits in the range of your encoding scheme (i.e. fp16 requires eps >= 6e-8).
    elementwise_scale : bool
        Include a learned scaling term after normalization?
    elementwise_shift : bool
        Include a learned bias term after normalization?
    use_mean : bool
        Recenter inputs around zero before normalizing, or just rescale?
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_scale=True,
        elementwise_shift=False,
        use_mean=False,
        use_high_precision_pow=False,
    ):
        super(LayerNormParameterized, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_scale = elementwise_scale
        self.elementwise_shift = elementwise_shift
        self.use_mean = use_mean
        self.use_high_precision_pow = use_high_precision_pow

        if self.elementwise_scale:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        # else:
        #     self.register_parameter("weight", None)
        if self.elementwise_shift:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        # else:
        #     self.register_parameter("bias", None)
        self.reset_params()

    def reset_params(self):
        if self.elementwise_scale:
            self.weight.data.fill_(1)
        if self.elementwise_shift:
            self.bias.data.zero_()

    def forward(self, x, residual=None):
        original_x = x
        # UNCOMMENT FOR old implementation
        if self.use_mean:
            x = x - x.mean(-1, keepdim=True)
        # x = F.normalize(x, dim=-1)*math.sqrt(x.size(-1))
        xf = x
        if self.use_high_precision_pow:
            xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        x = xf.type_as(x)
        if self.elementwise_scale:
            x = self.weight * x
        if self.elementwise_shift:
            x = x + self.bias

        if residual is None:
            residual = x
        else:
            residual = residual + x

        return x, residual
        # UNCOMMENT FOR layernorm kernel
        # original_shape = original_x.shape
        # normed_hidden_states, res, *rest = dropout_layer_norm.dropout_add_ln_fwd(
        #     original_x.view(-1, original_shape[2]),
        #     None if residual is None else residual.view(-1, original_shape[2]),
        #     self.weight,
        #     None,
        #     None,
        #     None,
        #     None,
        #     None,
        #     0.0,
        #     self.eps,
        #     1.0,
        #     0,
        #     None,
        #     False,
        #     True,  # Activate RMSNorm
        # )
        # normed_hidden_states = normed_hidden_states.view(*original_shape)
        # if res is None:
        #     res = original_x
        #
        # return normed_hidden_states, res
