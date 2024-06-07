import math
import torch
from scipy.linalg import hadamard

dtype   =torch.float32 #16 #.float16
qdtype  =torch.float32 #16 #.int8
accdtype=torch.float32 #16 #.int16

def quantize(weight, qdtype):
    # TODO: make sure it works for all dtypes
    if qdtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        scale_max = min(torch.finfo(qdtype).max, -torch.finfo(qdtype).min)
    elif qdtype in [torch.int8, torch.int16]:
        scale_max = min(torch.iinfo(qdtype).max, -torch.iinfo(qdtype).min)
    elif qdtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
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

ln_attn = ['error'] * 32
ln_ffn = ['error'] * 32

rots = []
sizes = [4096, 128, 128, 11008]
swap = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
for size in sizes:
    # tiled = diag_tile_block(swap, size // 2)
    # rots.append((tiled, tiled))
    rots.append((torch.eye(size, dtype=dtype), torch.eye(size, dtype=dtype)))
    # rots.append(get_almost_hadamard(size))
index = 0
rots[1] = (diag_tile_block(rots[1][0], 32), diag_tile_block(rots[1][1], 32))
# tiled = diag_tile_block(swap, sizes[index] // 2)
# rots[index] = (tiled, tiled)

for i in range(len(rots)):
    rots[i] = (rots[i][0].cuda(), rots[i][1].cuda()) # TODO: remove

def weight_check(key_steps, targets):
    if isinstance(targets, str):
        targets = [targets]
    return len(set(key_steps).intersection(targets)) > 0

def get_pre_rot(key_steps):
    if weight_check(key_steps, ['wg', 'w1']):
        return rots[0][1] @ torch.diag(ln_ffn[int(key_steps[1])].type(dtype)) #.view(1, -1).type(dtype) # [2] is the block number
    if weight_check(key_steps, 'w2'):
        return rots[3][1]
    if weight_check(key_steps, ['query', 'key', 'value']):
        return rots[0][1] @ torch.diag(ln_attn[int(key_steps[1])].type(dtype)) #.view(1, -1).type(dtype) # will brodcast across columns, equivalent of a diagonal matrix
    if weight_check(key_steps, 'dense'):
        return rots[1][1]
    else:
        return None

def get_post_rot(key_steps):
    if weight_check(key_steps, 'dense'):
        return rots[0][0]
    if weight_check(key_steps, 'value'):
        return rots[1][0]
    if weight_check(key_steps, 'w2'):
        return rots[0][0]
    else:
        return None

