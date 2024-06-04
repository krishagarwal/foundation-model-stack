import math
import torch
from scipy.linalg import hadamard

dtype   =torch.float16 #.float16
qdtype  =torch.float16 #.int8
accdtype=torch.float16 #.int16

def quantize(weight, qdtype):
    # TODO: make sure it works for all dtypes
    if qdtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        scale_max = min(torch.finfo(qdtype).max, -torch.finfo(qdtype).min)
    elif qdtype in [torch.int8, torch.int16]:
        scale_max = min(torch.iinfo(qdtype).max, -torch.iinfo(qdtype).min)
    elif qdtype in [torch.float16, torch.float32, torch.float64]:
        return weight.type(qdtype), torch.tensor(1, dtype=dtype), torch.tensor(0, dtype=dtype)
    else:
        raise ValueError("type not supported :(")

    minv = weight.min()
    maxv = weight.max()

    avg = (maxv + minv) / 2
    mag = maxv - avg

    offset = avg
    remaining = weight - offset

    scale = mag / scale_max
    remaining = remaining / (scale)

    return remaining.type(qdtype), scale, offset

def diag_tile_block(block, reps):
    assert block.shape[-1] == block.shape[-2]
    row = torch.nn.functional.pad(block, (0, block.shape[-1] * (reps - 1), 0, 0))
    return torch.concat(
        [torch.roll(row, block.shape[-1] * i, 1) for i in range(0, reps)]
    )

def get_almost_hadamard(size: int):
    tile_size = 1
    while size % tile_size == 0:
        tile_size *= 2
    tile_size //= 2
    tile_count = size // tile_size

    temp = torch.tensor(hadamard(tile_size), dtype=torch.float32) / torch.sqrt(torch.tensor(tile_size))
    m = diag_tile_block(temp, tile_count)

    m_inv = m.T
    m = m.type(dtype)
    m_inv = m_inv.type(dtype)
    return m, m_inv

rots = []
sizes = [4096, 4096, 128, 11008]
swap = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
for size in sizes:
    tiled = diag_tile_block(swap, size // 2)
    rots.append((tiled, tiled))
    # rots.append((torch.eye(size, dtype=dtype), torch.eye(size, dtype=dtype)))

def get_pre_rot(key_steps):
    if key_steps[-2] in ['gate_proj', 'up_proj']:
        return rots[0][1]
    if key_steps[-2] == 'down_proj':
        return rots[3][1]
    if key_steps[-2] in ['q_proj', 'k_proj', 'v_proj']:
        return rots[0][1]
    if key_steps[-2] == 'o_proj':
        return rots[1][1]
    else:
        return None

def get_post_rot(key_steps):
    if key_steps[-2] == 'o_proj':
        return rots[0][0]
    if key_steps[-2] == 'v_proj':
        return rots[1][0]
    if key_steps[-2] == 'down_proj':
        return rots[0][0]
    else:
        return None

