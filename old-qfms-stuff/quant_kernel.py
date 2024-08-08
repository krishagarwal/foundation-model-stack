# From https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

import math
import torch

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.get_current_target()[0] == "cuda"
    # return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    # BLOCK_SIZE_M = [16]
    # BLOCK_SIZE_N = [16]
    # BLOCK_SIZE_K = [32, 64, 256, 1024]
    # GROUP_SIZE_M = [1]
    # NUM_STAGES = [2, 4, 8]
    # NUM_WARPS = [2, 4, 8]

    # from itertools import product
    # confs = product(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, NUM_STAGES, NUM_WARPS)
    # confs = [triton.Config(
    #     {'BLOCK_SIZE_M': conf[0], 'BLOCK_SIZE_N': conf[1], 'BLOCK_SIZE_K': conf[2], 'GROUP_SIZE_M': conf[3]}, num_stages=conf[4], num_warps=conf[5]) 
    #     for conf in confs]
    
    # return confs

    return [
        triton.Config({'BLOCK_SIZE_M': 1,}, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 2,}, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 4,}, num_warps=16),
        # triton.Config({'BLOCK_SIZE_M': 8,}, num_warps=32),

        # triton.Config({'BLOCK_SIZE_M': 1,}, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 2,}, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 4,}, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 8,}, num_warps=2),

        # triton.Config({'BLOCK_SIZE_M': 1,}, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 2,}, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 4,}, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 8,}, num_warps=4),

        # # triton.Config({'BLOCK_SIZE_M': 1,}, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 2,}, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 4,}, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 8,}, num_warps=8),

        # # triton.Config({'BLOCK_SIZE_M': 1,}, num_warps=16),
        # # triton.Config({'BLOCK_SIZE_M': 2,}, num_warps=16),
        # triton.Config({'BLOCK_SIZE_M': 4,}, num_warps=16),
        # triton.Config({'BLOCK_SIZE_M': 8,}, num_warps=16),

        # # triton.Config({'BLOCK_SIZE_M': 1,}, num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 2,}, num_warps=32),
        # # triton.Config({'BLOCK_SIZE_M': 4,}, num_warps=32),
        # triton.Config({'BLOCK_SIZE_M': 8,}, num_warps=32),
    ]


def get_autotune_config():
    # if is_cuda():
        return get_cuda_autotune_config()
    # else:
    #     return get_hip_autotune_config()



@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N'],
    # restore_value=['c_ptr'],
)
@triton.jit
def quant(
    a_ptr, c_ptr, o_ptr, s_ptr,  #
    M, N: tl.constexpr,  #
    a_stride0, a_stride1,  #
    c_stride0, c_stride1,  #
    o_stride0, o_stride1,  #
    s_stride0, s_stride1,  #
    clip_ratio,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
):
    # https://arxiv.org/pdf/2402.00025
    pid = tl.program_id(0)

    # bits=8
    dim=1

    max_qint = 255 #2 ** bits - 1
    min_qint = 0

    offs_m = (pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = tl.arange(0, N)
    mask_m = offs_m < M

    a_ptrs = a_ptr + (offs_m[:, None] * a_stride0 + offs_n[None, :] * a_stride1)

    a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
    a_max_m = tl.max(a)#, axis=dim)
    a_min_m = tl.min(a)#, axis=dim)
    a_max_m = tl.maximum(a_max_m[:, None], 0)
    a_min_m = tl.minimum(a_min_m[:, None], 0)
    is_zero = (a_max_m == 0) & (a_min_m == 0)
    a_max_m = tl.where(is_zero, 1, a_max_m)
    a_min_m = tl.where(is_zero, 1, a_min_m)
    mag = a_max_m - a_min_m
    mag *= clip_ratio
    scale = mag / max_qint
    offset = tl.div_rn(-a_min_m, scale) # tl.ceil(-a_min_m / scale)
    weight = a / scale + offset
    # weight = tl.clamp(weight, min=min_qint, max=max_qint).round()
    weight = tl.where(weight < min_qint, min_qint, weight)
    weight = tl.where(weight > max_qint, max_qint, weight)
    weight = tl.cast(tl.div_rn(weight, 1), tl.uint8)#, fp_downcast_rounding="rtne")

    c_ptrs = c_ptr + (offs_m[:, None] * c_stride0 + offs_n[None, :] * c_stride1)
    s_ptrs = s_ptr + (offs_m * s_stride0)
    o_ptrs = o_ptr + (offs_m * o_stride0)

    tl.store(c_ptrs, weight, mask=mask_m[:, None])
    tl.store(s_ptrs, scale, mask=mask_m)
    tl.store(o_ptrs, offset, mask=mask_m)


torch.library.define("quantized::quant", "(Tensor a, float clip_ratio) -> (Tensor, Tensor, Tensor)")

@torch.library.impl("quantized::quant", "cuda")
def quant_fp16_int8(a, clip_ratio):
    # print("yeet")
    # return (torch.eye(1), torch.eye(1), torch.eye(1))
    out_shape = a.shape
    # a = a.view(-1, a.shape[-1])
    assert a.stride(-1) == 1
    # Check constraints.
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    M, N = 1, a.shape[-1]
    for a_shape in a.shape[:-1]:
        M *= a_shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.uint8) # must be zeros for the split_k kernel
    o = torch.empty((M, 1), device=a.device, dtype=torch.float16) # must be zeros for the split_k kernel
    s = torch.empty((M, 1), device=a.device, dtype=torch.float16) # must be zeros for the split_k kernel
    # 1D launch kernel where each block gets its own program.
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),) # TODO: fix

    quant[grid](
        a, c, o, s,  #
        M, N,  #
        N, 1,# a.stride(0), a.stride(1),  #
        c.stride(0), c.stride(1),  #
        o.stride(0), o.stride(1),  #
        s.stride(0), s.stride(1),  #
        clip_ratio,
    )
    c = c.view(out_shape)
    o = o.view((*out_shape[:-1], 1))
    s = s.view((*out_shape[:-1], 1))
    return c, s, o

@torch.library.impl_abstract("quantized::quant")
def quant_fp16_int8_faketensor(a: torch.Tensor, clip_ratio: float):
    return (
        torch.empty(a.shape, dtype=torch.int32, device=a.device), 
        torch.empty((*a.shape[:-1], 1), dtype=torch.int32, device=a.device), 
        torch.empty((*a.shape[:-1], 1), dtype=torch.int32, device=a.device)
    )




@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N'],
    # restore_value=['c_ptr'],
)
@triton.jit
def quant_sym(
    a_ptr, c_ptr, s_ptr,  #
    M, N, N_2: tl.constexpr,  #
    a_stride0, a_stride1,  #
    c_stride0, c_stride1,  #
    s_stride0, s_stride1,  #
    clip_ratio,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
):
    # https://arxiv.org/pdf/2402.00025
    pid = tl.program_id(0)

    # bits=8
    # dim=1

    max_qint = 127 #2 ** bits - 1
    min_qint = -128

    offs_m = (pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = tl.arange(0, N_2)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    a_ptrs = a_ptr + (offs_m[:, None] * a_stride0 + offs_n[None, :] * a_stride1)

    a = tl.load(a_ptrs, mask=mask, other=0.0)
    mag = tl.max(a.abs())
    mag *= clip_ratio
    is_zero = mag == 0
    scale = tl.where(is_zero, 1, mag / max_qint)
    weight = a / scale
    weight = tl.where(weight < min_qint, min_qint, weight)
    weight = tl.where(weight > max_qint, max_qint, weight)
    weight = tl.cast(tl.div_rn(weight, 1), tl.int8)#, fp_downcast_rounding="rtne")

    c_ptrs = c_ptr + (offs_m[:, None] * c_stride0 + offs_n[None, :] * c_stride1)
    s_ptrs = s_ptr + (offs_m * s_stride0)

    tl.store(c_ptrs, weight, mask=mask)
    tl.store(s_ptrs, scale, mask=mask_m)


torch.library.define("quantized::quant_sym", "(Tensor a, float clip_ratio) -> (Tensor, Tensor)")

@torch.library.impl("quantized::quant_sym", "cuda")
def quant_sym_fp16_int8(a, clip_ratio):
    # print("yeet")
    # return (torch.eye(1), torch.eye(1), torch.eye(1))
    out_shape = a.shape
    # a = a.view(-1, a.shape[-1])
    assert a.stride(-1) == 1
    # Check constraints.
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    M, N = 1, a.shape[-1]
    for a_shape in a.shape[:-1]:
        M *= a_shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.int8) # must be zeros for the split_k kernel
    s = torch.empty((M, 1), device=a.device, dtype=torch.float16) # must be zeros for the split_k kernel
    # 1D launch kernel where each block gets its own program.
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),) # TODO: fix
    N_2 = 2 ** (math.ceil(math.log2(N)))
    quant_sym[grid](
        a, c, s,  #
        M, N, N_2,  #
        N, 1,# a.stride(0), a.stride(1),  #
        c.stride(0), c.stride(1),  #
        s.stride(0), s.stride(1),  #
        clip_ratio,
    )
    c = c.view(out_shape)
    s = s.view((*out_shape[:-1], 1))
    return c, s

@torch.library.impl_abstract("quantized::quant_sym")
def quant_sym_fp16_int8_faketensor(a: torch.Tensor, clip_ratio: float):
    return (
        torch.empty(a.shape, dtype=torch.int32, device=a.device), 
        torch.empty((*a.shape[:-1], 1), dtype=torch.int32, device=a.device)
    )


# if __name__ == "__main__":
#     a = torch.randn((8, 128), device='cuda', dtype=torch.float16)
#     w, s, o = quant_fp16_int8(a)
#     print(a)
#     print(w)
#     print(s)
#     print(o)