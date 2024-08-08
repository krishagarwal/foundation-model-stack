#!/usr/bin/env python
"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team, AMD ML Frameworks Triton team

Features supported:

1) Fwd with causal masking
2) Any sequence lengths without padding (currently fwd kernel only)
3) Support for different sequence lengths for q and k
4) Nested tensor API currently does not support dropout or bias.

Not currently supported:

1) Non power of two head dims

"""

import argparse
import pytest
import random
import sys
import torch

import triton
import triton.language as tl

torch_dtype:tl.constexpr = torch.float16

# TORCH_HAS_FP8E5 = hasattr(torch, 'float8_e5m2fnuz')
# if TORCH_HAS_FP8E5:
#     torch_dtype:tl.constexpr = torch.float8_e5m2fnuz

def check_args(q, k, v, o):
    assert q.dim() == k.dim() and q.dim() == v.dim()
    assert q.dim() == 4
    batch, nheads_q, seqlen_q, head_size = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    assert k.shape == v.shape
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
    # TODO: Change assert if we support qkl f8 and v f16
    # assert q.dtype == k.dtype and q.dtype == v.dtype
    assert head_size <= 256
    assert o.shape == q.shape
    assert (nheads_q % nheads_k) == 0

@triton.jit
def cdiv_fn(x,y):
    return (x + y - 1) // y

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

@triton.jit
def load_fn(block_ptr, first, second, pad):
    if first and second:
        tensor = tl.load(block_ptr, boundary_check=(0,1), padding_option=pad)
    elif first:
        tensor = tl.load(block_ptr, boundary_check=(0,), padding_option=pad)
    elif second:
        tensor = tl.load(block_ptr, boundary_check=(1,), padding_option=pad)
    else:
        tensor = tl.load(block_ptr)
    return tensor

@triton.jit
def print_gpu(prefix, val=None):
    if (tl.program_id(0) == 0) and ((tl.program_id(1) == 0) and (tl.program_id(2) == 0)):
        if val is not None:
            tl.device_print(prefix, val)
        else:
            tl.device_print(prefix)


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m,
    actual_seqlen_k,
    actual_seqlen_q,
    block_min, block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    PADDED_HEAD: tl.constexpr
):
    # loop over k, v, and update accumulator
    for start_n in range (block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        k = load_fn(K_block_ptr, PADDED_HEAD, MASK_STEPS and (n_extra_tokens != 0), "zero")
        if PRE_LOAD_V:
            v = load_fn(V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None,:]
                mask = size_n < boundary_m[:,None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        # -- compute qk ----
        k = k.to(tl.float16)
        qk += tl.dot(q, k)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk,1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(V_block_ptr, MASK_STEPS and (n_extra_tokens != 0), PADDED_HEAD, "zero")
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        v = v.to(tl.float16)
        acc += tl.dot(p.to(tl.float16), v)
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i

@triton.autotune(
   configs=[
    #    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'PRE_LOAD_V': False}, num_stages=1, num_warps=8),
    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'PRE_LOAD_V': False}, num_stages=1, num_warps=4),
    # #    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'PRE_LOAD_V': False}, num_stages=1, num_warps=8),
    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': True}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'PRE_LOAD_V': False}, num_stages=1, num_warps=8),
    #    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': False}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'PRE_LOAD_V': False}, num_stages=1, num_warps=8),
    #    # TODO: This config fails with head_size not pow2 with data mismatches. Check why.
    # #    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'PRE_LOAD_V': False}, num_stages=1, num_warps=4),
    #    triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'PRE_LOAD_V': False}, num_stages=1, num_warps=4),


    # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512, 'PRE_LOAD_V': False}, num_stages=1 , num_warps=32),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'PRE_LOAD_V': False}, num_stages=2 , num_warps=32),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'PRE_LOAD_V': False}, num_stages=4 , num_warps=32),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  64, 'PRE_LOAD_V': False}, num_stages=8 , num_warps=32),
    # # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=16, num_warps=8),
    # # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  16, 'PRE_LOAD_V': False}, num_stages=32, num_warps=8),

    # # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512, 'PRE_LOAD_V': False}, num_stages=1 , num_warps=4),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'PRE_LOAD_V': False}, num_stages=1 , num_warps=16),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'PRE_LOAD_V': False}, num_stages=2 , num_warps=16),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  64, 'PRE_LOAD_V': False}, num_stages=4 , num_warps=16),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=8 , num_warps=16),
    # # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  16, 'PRE_LOAD_V': False}, num_stages=16, num_warps=4),

    # # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512, 'PRE_LOAD_V': False}, num_stages=1 , num_warps=4),
    # # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'PRE_LOAD_V': False}, num_stages=1 , num_warps=4),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'PRE_LOAD_V': False}, num_stages=1 , num_warps=8),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  64, 'PRE_LOAD_V': False}, num_stages=2 , num_warps=8),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=4 , num_warps=8),
    # # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  16, 'PRE_LOAD_V': False}, num_stages=16, num_warps=4),

    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=4 , num_warps=8),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=4 , num_warps=16),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=4 , num_warps=32),

    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=8 , num_warps=8),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=8 , num_warps=16),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=8 , num_warps=32),

    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=16, num_warps=8),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=16, num_warps=16),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=16, num_warps=32),


    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': True}, num_stages=4 , num_warps=8),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': True}, num_stages=4 , num_warps=16),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': True}, num_stages=4 , num_warps=32),

    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': True}, num_stages=8 , num_warps=8),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': True}, num_stages=8 , num_warps=16),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': True}, num_stages=8 , num_warps=32),

    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': True}, num_stages=16, num_warps=8),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': True}, num_stages=16, num_warps=16),
    # triton.Config({'BLOCK_M': 16, 'BLOCK_N':  32, 'PRE_LOAD_V': True}, num_stages=16, num_warps=32),


    triton.Config({'BLOCK_M': 1, 'BLOCK_N':  32, 'PRE_LOAD_V': False}, num_stages=4 , num_warps=8),
    
   ],
   key=['IS_CAUSAL', 'BLOCK_DMODEL'],
#    use_cuda_graph=True,
)
@triton.jit
def attn_fwd(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    HQ: tl.constexpr, HK:tl.constexpr,
    ACTUAL_BLOCK_DMODEL:tl.constexpr,
    MAX_SEQLENS_Q:tl.constexpr, MAX_SEQLENS_K:tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    seqlen_q = MAX_SEQLENS_Q
    seqlen_k = MAX_SEQLENS_K

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if (IS_CAUSAL):
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.
        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = cdiv_fn(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q,
            BLOCK_N
        )
        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)
        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            o_offset = off_z * stride_oz + off_h_q * stride_oh
            O_block_ptr = tl.make_block_ptr(
                base=Out + o_offset,
                shape=(seqlen_q, BLOCK_DMODEL),
                strides=(stride_om, stride_on),
                offsets=(start_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_DMODEL),
                order=(1, 0)
            )
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
            # We still need to write 0s to the result
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,1))
            return

    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    need_padding = False
    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        need_padding = True
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        need_padding = True
        n_extra_tokens = seqlen_k % BLOCK_N
    PADDED_HEAD:tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

    # Compute pointers for all the tensors used in this kernel.
    q_offset = off_z * stride_qz +  off_h_q * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    k_offset = off_z * stride_kz + off_h_k * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(ACTUAL_BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    v_offset = off_z * stride_vz + off_h_k * stride_vh
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(seqlen_k, ACTUAL_BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
    # have native e^x support in HW.
    qk_scale = sm_scale * 1.44269504089
    # Q is loaded once at the beginning and shared by all N blocks.
    q = load_fn(Q_block_ptr, True, PADDED_HEAD, "zero")
    q = (q * qk_scale).to(Q_block_ptr.type.element_ty)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, seqlen_k, seqlen_q,
            # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
            block_min, block_max, 0, 0, 0,
            # IS_CAUSAL, ....
            False, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V, False, PADDED_HEAD
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    tl.debug_barrier()
    # Remaining blocks, if any, are full / not masked.
    if (masked_blocks > 0):
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        K_block_ptr = tl.advance(K_block_ptr, (0, n_full_blocks*BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (n_full_blocks*BLOCK_N, 0))
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, V_block_ptr,
            start_m, seqlen_k, seqlen_q,
            block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens,
            IS_CAUSAL, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V, True, PADDED_HEAD
        )
    # epilogue
    acc = acc / l_i[:, None]
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    acc = acc.to(Out.type.element_ty)
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL,), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
    # write back O
    o_offset = off_z * stride_oz + off_h_q * stride_oh
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(seqlen_q, ACTUAL_BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # Need boundary check on this to make sure the padding from the
    # Q and KV tensors in both dims are not part of what we store back.
    # TODO: Do the boundary check optionally.
    tl.store(O_block_ptr, acc, boundary_check=(0,1))

empty = torch.empty(128, device="cuda")


torch.library.define("dequant::attn", "(Tensor q, Tensor k, Tensor v, bool is_causal_mask) -> Tensor")

@torch.library.impl("dequant::attn", "cuda")
def forward(q, k, v, causal):
    o = torch.empty_like(q)
    check_args(q, k, v, o)
    batch, nheads_q, seqlen_q, head_size = q.shape
    _, nheads_k, seqlen_k, _ = k.shape
    q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
    k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
    v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
    o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))

    sm_scale = head_size ** -0.5

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)

    grid = lambda META: (
        triton.cdiv(q.shape[-2], META['BLOCK_M']),
        nheads_q,
        batch
    )


    attn_fwd[grid](
        q, k, v, sm_scale, o,
        *q_strides, *k_strides, *v_strides, *o_strides,
        HQ=nheads_q, HK=nheads_k,
        ACTUAL_BLOCK_DMODEL=head_size,
        MAX_SEQLENS_Q=seqlen_q,
        MAX_SEQLENS_K=seqlen_k,
        IS_CAUSAL=causal,
        BLOCK_DMODEL=padded_d_model,
        BATCH_SIZE= q.shape[0]
    )
    return o

attention = forward

@torch.library.impl_abstract("dequant::attn")
def fake_forward(q, k, v, causal):
    bs, nheads, seq_len, _ = q.shape
    emb_v_per_head = v.shape[-1]
    return torch.empty((bs, nheads, seq_len, emb_v_per_head), dtype=q.dtype, device=q.device)


def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype):
    torch.manual_seed(20)

    # Initialize q, k, v
    q = torch.randn((Z, HQ, N_CTX_Q, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((Z, HK, N_CTX_K, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((Z, HK, N_CTX_K, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    return q, k, v




@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD',
                         [(4, 48, 24, 1024, 1024, 64),
                          (1, 24, 6, 8192, 8192, 64),
                          (1, 4, 2, 16384, 16384, 128),
                          (2, 16, 4, 1020, 987, 128),
                          (2, 16, 4, 15498, 2, 128),
                          (2, 16, 2, 7, 16219, 64),
                          (4, 48, 12, 1, 1, 64),
                          (4, 48, 48, 1, 1, 128),
                          (4, 48, 24, 3, 3, 128),
                          (4, 48, 48, 1001, 990, 64),
                          (1, 8, 8, 8081, 7099, 64),
                          (1, 4, 4, 16330, 15989, 128),
                          (4, 4, 1, 1024, 1024, 33),
                          (4, 4, 2, 65, 1018, 65),
                          (4, 4, 4, 128, 128, 65),
                          (4, 4, 4, 113, 123, 1),
                          ])
@pytest.mark.parametrize('causal', [True])
def test_op_fwd(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q, k, v = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype)

    # if TORCH_HAS_FP8E5:
    #     q = q.to(torch_dtype)
    #     k = k.to(torch_dtype)
    o = torch.empty_like(q)

    # triton implementation
    tri_out = attention(q, k, v, o, causal)

    # Replicate K and V if using MQA/GQA
    if HQ != HK:
        k = k.view(
                k.shape[0], k.shape[1], -1, k.shape[2], k.shape[3]).expand(
                    -1, -1, HQ // HK, -1, -1).reshape(k.shape[0], -1, k.shape[2], k.shape[3])
        v = v.view(
                v.shape[0], v.shape[1], -1, v.shape[2], v.shape[3]).expand(
                    -1, -1, HQ // HK, -1, -1).reshape(v.shape[0], -1, v.shape[2], v.shape[3])
    
    sm_scale = D_HEAD ** -0.5
    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), 
                          diagonal=N_CTX_K-N_CTX_Q)
        scores[:, :, mask==0] = float("-inf")

    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
        # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
        # this by converting the NaNs to 0s, which is what they should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask==1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


def nonvarlen_benchmark_configs():
    configs=[(16, 16, 16, 1024, 1024),
            (8, 16, 16, 2048, 2048),
            (4, 16, 16, 4096, 4096),
            (2, 16, 16, 8192, 8192),
            (1, 16, 16, 16384, 16384),
            (2, 48, 48, 1024, 1024),
            (2, 48, 48, 2048, 1024),
            (2, 48, 48, 4096, 8192),
            (2, 48, 48, 8192, 4096),
            (2, 48, 48, 16384, 8192),
            (8, 16, 16, 1989, 15344),
            (4, 16, 16, 4097, 163),
            (2, 16, 16, 8122, 2159),
            (1, 16, 16, 16281, 7),
            (2, 48, 48, 1021, 1020),
            (2, 48, 48, 2001, 2048),
            (2, 48, 48, 3996, 9639),
            (2, 48, 48, 8181, 1021),
            ]
    return configs

def run_benchmark(custom):

    args = parse_args()
    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    mode = 'fwd'
    x_names=['BATCH', 'HQ', 'HK', 'N_CTX_Q', 'N_CTX_K']
    causal = args.causal
    configs = []
    if custom:
        x_vals_list=[(args.b, args.hq, args.hk, args.sq, args.sk)]
    else:
        x_vals_list = nonvarlen_benchmark_configs()
    print_time = args.return_time
    line_names = 'Time (ms)' if print_time else 'TFLOPS'
    configs.append(triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg='provider',
        line_vals=['triton'],
        line_names=[line_names],
        styles=[('red', '-')],
        ylabel='ms',
        plot_name=f'fused-attention-{mode}-d{head_size}',
        args={
            'D_HEAD': head_size,
            'dtype': dtype,
            'causal': causal,
            'mode': mode})
    )

    @triton.testing.perf_report(configs)
    def bench_flash_attention(
        BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, causal, mode, provider, device="cuda"
    ):
        assert mode in ["fwd"]
        warmup = 25
        rep = 100

        flops_per_matmul = 0
        q, k, v = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype)
        flops_per_matmul = 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * D_HEAD
        o = torch.empty_like(q)
        fn = lambda: attention(q, k, v, o, causal)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        total_flops = 2 * flops_per_matmul
        # TODO: This needs to be fixed for unequal Q/K seqlens
        if causal:
            total_flops *= 0.5
        if print_time:
            return ms
        else:
            return total_flops / ms * 1e-9

    bench_flash_attention.run(save_path=".", print_data=True)

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark FlashAttention",
        allow_abbrev=False,
    )
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-causal", action='store_true', default=False)
    parser.add_argument("-dtype", default='fp16')
    parser.add_argument("-return_time", action='store_true', default=False)
    return parser.parse_args()

arg_to_torch_dtype = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32
}

def main():
    args = parse_args()
    custom_config = False
    if args.b or args.hq or args.hk or args.sq or args.sk or args.d:
        custom_config = True
        assert args.b and args.hq and args.sq and args.d, \
               "If custom config is specified, please provide \
                all of batch, number of Q heads, Q sequence length \
                and head size."

    assert args.dtype in arg_to_torch_dtype, \
           "Only fp16, bf16 and f32 types currently supported."

    run_benchmark(custom_config)

if __name__ == '__main__':
    sys.exit(main())