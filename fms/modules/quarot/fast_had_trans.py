import math
import scipy.linalg
import torch
import triton
import triton.language as tl
import scipy
import itertools
from cuda import example
from fast_hadamard_transform import hadamard_transform
# from . import utils

def is_cuda():
    # return triton.runtime.driver.active.get_current_target().backend == "cuda"
    return triton.runtime.driver.active.get_current_target()[0] == "cuda"
    # return isinstance(triton.runtime.driver, triton.runtime.CudaDriver) #triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3,
                      num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_stages=4,
        #               num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2},
            num_warps=4, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2},
            num_warps=8, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 2},
            num_warps=8, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 3},
            num_warps=4, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 8},
            num_warps=4, num_stages=0),
    ]

def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()



# import os
# os.environ["TRITON_INTERPRET"] = "1"

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
# @triton.autotune(
#     configs=get_autotune_config(),
#     key=['M', 'N'],
#     restore_value=['a_ptr']
# )
@triton.jit
def triton_trans(a_ptr, M: tl.constexpr, N: tl.constexpr, stride_m, stride_n, h: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    m_range = tl.arange(0, BLOCK_SIZE_M) + pid_0 * BLOCK_SIZE_M
    n_range = tl.arange(0, BLOCK_SIZE_N) + pid_1 * BLOCK_SIZE_N

    my_id = (m_range % h) + (m_range // h) * h * 2
    my_col = n_range
    my_col_mask = my_col < N

    idx1 = (my_id * stride_m).expand_dims(1) + (my_col * stride_n).expand_dims(0)
    idx1_mask = tl.broadcast_to(my_col_mask.expand_dims(0), BLOCK_SIZE_M, BLOCK_SIZE_N)
    x = tl.load(a_ptr + idx1, idx1_mask)
    y = tl.load((a_ptr + h * stride_m) + idx1, idx1_mask)
    tl.store(a_ptr + idx1, (x + y) * 0.7071067811865475, idx1_mask)
    tl.store((a_ptr + h * stride_m) + idx1, (x - y) * 0.7071067811865475, idx1_mask)


@triton.jit
def triton_below_1k_trans(a_ptr, M: tl.constexpr, N: tl.constexpr, stride_m, stride_n, had_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    m_range = tl.arange(0, BLOCK_SIZE_M) + pid_0 * BLOCK_SIZE_M
    n_range = tl.arange(0, BLOCK_SIZE_N) + pid_1 * BLOCK_SIZE_N

    my_col = n_range
    my_col_mask = my_col < N
    idx1_mask = tl.broadcast_to(my_col_mask.expand_dims(0), BLOCK_SIZE_M, BLOCK_SIZE_N)

    # my_chunk = tl.load(a_ptr + (m_range * stride_m)[:, None] + (n_range * stride_n)[None, :])

    h = 1
    while h < had_size:
        my_id = (m_range % h) + (m_range // h) * h * 2
        idx1 = (my_id * stride_m).expand_dims(1) + (my_col * stride_n).expand_dims(0)
        x = tl.load(a_ptr + idx1, idx1_mask)
        y = tl.load((a_ptr + h * stride_m) + idx1, idx1_mask)
        tl.store(a_ptr + idx1, (x + y) * tl.rsqrt(2.0), idx1_mask)
        tl.store((a_ptr + h * stride_m) + idx1, (x - y) * tl.rsqrt(2.0), idx1_mask)
        # x = my_chunk[0]
        # x = tl.load(my_chunk + tl.arange(0, 1))
        # y = my_chunk[idx1 + h]
        h *= 2
        triton.language.debug_barrier()
    
    # tl.store(a_ptr + (m_range * stride_m)[:, None] + (n_range * stride_n)[None, :], my_chunk)

# requires matrix dim 0 is power of 2 and had_size is power of 2
def triton_fast_had_2d(a, had_size):
    grid = lambda META: (triton.cdiv(a.shape[0], 2 * META['BLOCK_SIZE_M']), triton.cdiv(a.shape[1], META['BLOCK_SIZE_N']), )
    if had_size is None:
        had_size = a.shape[0]

    ELEMENTS_PER_THREAD = 4
    ELEMENTS_PER_BLOCK = 256
    block_size_m = min(had_size // 2, ELEMENTS_PER_BLOCK)
    block_size_n = ELEMENTS_PER_BLOCK // block_size_m
    h = 2 * block_size_m
    triton_below_1k_trans[grid](a, a.shape[0], a.shape[1], a.stride(0), a.stride(1), h, BLOCK_SIZE_M=block_size_m, BLOCK_SIZE_N=block_size_n, num_warps=ELEMENTS_PER_BLOCK // 32 // ELEMENTS_PER_THREAD)

    # h = 1

    while h < had_size:
        triton_trans[grid](a, a.shape[0], a.shape[1], a.stride(0), a.stride(1), h, BLOCK_SIZE_M=16, BLOCK_SIZE_N=32)
        h *= 2
    return a

    # a = hadamard_transform(a.T.view(-1, had_size)).view(*a.shape[::-1]).T / math.sqrt(a.shape[0])
    return a

cached_pointers = {}
def fast_had_2d_graph_wrapper(a, had_size=None, use_graph=False):
    if had_size is None:
        had_size = a.shape[0]

    if not use_graph:# or not utils.use_graph:
        return triton_fast_had_2d(a, had_size) # hadamard_transform(a.T.view(a.shape[1], -1, had_size)).view_as(a.T).T / math.sqrt(had_size)

    a_key = (tuple(a.shape), had_size)
    if a_key not in cached_pointers:
        pointer = torch.empty_like(a)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            triton_fast_had_2d(a, had_size)
            # pointer.copy_(hadamard_transform(pointer.T.view(pointer.shape[1], -1, had_size)).view_as(pointer.T).T)
        cached_pointers[a_key] = (pointer, g)
        
    pointer, graph = cached_pointers[a_key]
    pointer.copy_(a)
    graph.replay()
    a.copy_(pointer / math.sqrt(had_size))
    return a


# TODO: remove
def right_had(a, had_size=None, use_graph=False):
    # return triton_fast_had(a.transpose(-2, -1), had_size=had_size).transpose(-2, -1)
    if len(a.shape) == 2:
        return fast_had_2d_graph_wrapper(a.T, had_size, use_graph).T
    flats = a.shape[:-1]
    a_flat = a.flatten(0, -2).T # don't flatten last dim
    res_flat = fast_had_2d_graph_wrapper(a_flat, had_size, use_graph)
    return (res_flat.T.unflatten(0, flats))

def triton_fast_had(a, had_size=None, use_graph=False):
    if len(a.shape) == 2:
        return fast_had_2d_graph_wrapper(a, had_size, use_graph)
    a = a.transpose(-2, -1)
    flats = a.shape[:-1]
    a_flat = a.flatten(0, -2).T # don't flatten last dim
    res_flat = fast_had_2d_graph_wrapper(a_flat, had_size, use_graph)
    return (res_flat.T.unflatten(0, flats)).transpose(-2, -1)

# import random
if __name__ == "__main__":
    # M, N = 128, 45
    # a = torch.ones((M, 32 * N), dtype=torch.float16).cuda()

    M, N = 256, 4096 * 16
    had_size = 256
    # a = torch.eye(128, dtype=torch.float16).cuda()
    a = torch.ones((M, N), dtype=torch.float16).cuda()

    # b = a.clone()
    
    # s = torch.cuda.Stream()
    # s.wait_stream(torch.cuda.current_stream())
    # with torch.cuda.stream(s):
    #     # x = a @ a.T
    #     # for i in range(10):
    #     triton_fast_had_2d(a)

    # torch.cuda.current_stream().wait_stream(s)
    # s.wait_stream(torch.cuda.current_stream())
    
    # # capture
    # g = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(g):
    #     # for i in range(10):
    #     triton_fast_had_2d(a)
    
    # # b = torch.empty_like(a)
    # # b.copy_(a)

    # for i in range(7):
    #     # b.copy_(a)
    #     g.replay()
    #     # print(a)
    
    # print(a)
    a_temp = torch.empty(a.shape[::-1]).cuda().to(torch.float16)
    a_temp.copy_(a.T)
    a = a_temp.T

    for _ in range(5):
        # a = hadamard_transform(a.T).T / math.sqrt(a.shape[0])
        # example.fast_had_trans(a.data_ptr(), a.shape[0], a.shape[1], a.stride(0), a.stride(1), had_size)
        a = fast_had_2d_graph_wrapper(a, had_size=had_size, use_graph=False)
    print(a)

@torch.compile()
def torch_fast_had(a: torch.Tensor, had_size=None):
    if had_size is None:
        had_size = a.shape[-2]
    h = 1
    while h < had_size:
        i_range = torch.arange(0, a.shape[-2] // (h * 2), device='cuda') * h * 2
        j_range = torch.arange(0, h, device='cuda')
        idxs = (i_range.view(-1, 1) + j_range.view(1, -1)).view(-1)
        x = a[idxs, :]
        y = a[idxs + h, :]
        a[idxs, :] = (x + y) * 0.7071067811865475
        a[idxs + h, :] = (x - y) * 0.7071067811865475
        h *= 2
    return a