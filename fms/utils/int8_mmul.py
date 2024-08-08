# this custom int8 matmul kernel is necessary since torch._scaled_mm has excessive size requirements causing 2x wasted work
# adapted from the split-k triton paper, but without split-k since we couldn't get a speedup
# TODO: consider adding split-k: https://arxiv.org/pdf/2402.00025

import torch

import triton
import triton.language as tl

def get_autotune_config():
    # BLOCK_SIZE_M = [16]
    # BLOCK_SIZE_N = [16]
    # BLOCK_SIZE_K = [32, 64, 256, 1024]
    # NUM_STAGES = [2, 4, 8]
    # NUM_WARPS = [2, 4, 8]
    # from itertools import product
    # confs = product(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, NUM_STAGES, NUM_WARPS)
    # confs = [triton.Config(
    #     {'BLOCK_SIZE_M': conf[0], 'BLOCK_SIZE_N': conf[1], 'BLOCK_SIZE_K': conf[2]}, num_stages=conf[3], num_warps=conf[4]) 
    #     for conf in confs]
    # return confs

    return [
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256 }, num_stages=11, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 1024}, num_stages=3,  num_warps=2),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_loop_count = tl.cdiv(K, BLOCK_SIZE_K)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    offs_k = (pid_k * K + tl.arange(0, BLOCK_SIZE_K))

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.int32)
    for k in range(0, k_loop_count):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K), other=0.0)
        # We accumulate along the K dimension.
        acc = tl.dot(a, b, acc)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, acc, mask=c_mask)

torch.library.define("quantized::int8_mmul", "(Tensor a, Tensor b) -> Tensor")

@torch.library.impl("quantized::int8_mmul", "cuda")
def int8_matmul(a, b):
    out_shape = (*a.shape[:-1], b.shape[-1])
    a = a.view(-1, a.shape[-1])
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    c = c.view(out_shape)
    return c

@torch.library.impl_abstract("quantized::int8_mmul")
def int8_mmul_faketensor(a: torch.Tensor, b: torch.Tensor):
    out_shape = (*a.shape[:-1], b.shape[-1])
    return torch.empty(out_shape, dtype=torch.int32, device=a.device)

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_int4_packed(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    k_loop_count = tl.cdiv(K, BLOCK_SIZE_K)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    offs_k = (pid_k * K + tl.arange(0, BLOCK_SIZE_K))
    offs_k_b = (pid_k * K // 2 + tl.arange(0, BLOCK_SIZE_K // 2))

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k_b[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    b_low = tl.broadcast_to(offs_k[:, None] % 2 == 1, (BLOCK_SIZE_K, BLOCK_SIZE_N))

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.int32)
    for k in range(0, k_loop_count):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k_b[:, None] < (K - k * BLOCK_SIZE_K) // 2), other=0.0)

        b = tl.view(tl.where(tl.view(b_low, (2, BLOCK_SIZE_K // 2, BLOCK_SIZE_N)),
                      tl.view((b << 4) >> 4, (1, BLOCK_SIZE_K // 2, BLOCK_SIZE_N)), 
                      tl.view((b >> 4),      (1, BLOCK_SIZE_K // 2, BLOCK_SIZE_N))), (BLOCK_SIZE_K, BLOCK_SIZE_N))
        # b = tl.view(tl.where(tl.view(b_low, (2, BLOCK_SIZE_K // 2, BLOCK_SIZE_N)),
        #               tl.view(b, (1, BLOCK_SIZE_K // 2, BLOCK_SIZE_N)), 
        #               tl.view(b,      (1, BLOCK_SIZE_K // 2, BLOCK_SIZE_N))), (BLOCK_SIZE_K, BLOCK_SIZE_N))

        b = b.to(tl.int8)

        # b = tl.view(tl.reshape(tl.view(b,
        #             (1, BLOCK_SIZE_K // 2, BLOCK_SIZE_N)),
        #         (2, BLOCK_SIZE_K // 2, BLOCK_SIZE_N)),
        #     (BLOCK_SIZE_K, BLOCK_SIZE_N))
        
        # b1, b2 = b >> 4, (b << 4) >> 4 # need left + right shift here for sign extension to work
        # b = tl.cat(b1, b2)
        # torch.cat((a1.unsqueeze(-1), a2.unsqueeze(-1)), -1).view(*a.shape[:-1], a.shape[-1] * 2) # TODO: consider not hardcoding -1 dim
        
        # We accumulate along the K dimension.
        acc = tl.dot(a, b, acc)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K // 2 * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, acc, mask=c_mask)

torch.library.define("quantized::int8_mmul_unpack_int4", "(Tensor a, Tensor b) -> Tensor")

@torch.library.impl("quantized::int8_mmul_unpack_int4", "cuda")
def int4_matmul(a, b):
    out_shape = (*a.shape[:-1], b.shape[-1])
    a = a.view(-1, a.shape[-1])
    # Check constraints.
    assert a.shape[1] == b.shape[0] * 2, "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape[0] * 2, b.shape[1]
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel_int4_packed[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    c = c.view(out_shape)
    return c

@torch.library.impl_abstract("quantized::int8_mmul_unpack_int4")
def int4_mmul_faketensor(a: torch.Tensor, b: torch.Tensor):
    out_shape = (*a.shape[:-1], b.shape[-1])
    return torch.empty(out_shape, dtype=torch.int32, device=a.device)