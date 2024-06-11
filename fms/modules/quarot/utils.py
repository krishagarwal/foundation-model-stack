import math
import torch
from scipy.linalg import hadamard
import random

dtype   =torch.float16 # .float16 #
qdtype  =torch.int8 # .float16 #
accdtype=torch.int32 # 16 # .float16 #

def quantize(weight, qdtype, device=None):
    if device is None:
        device = weight.device
    
    # TODO: make sure it works for all dtypes
    if qdtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        scale_max = torch.tensor([torch.finfo(qdtype).max, -torch.finfo(qdtype).min]).min().to(weight.device, dtype=dtype) # TODO: check if doing scale/offset calculations on gpu is optimal # TODO: find cleaner way to get min
    elif qdtype in [torch.int8, torch.int16]:
        scale_max = torch.tensor([torch.iinfo(qdtype).max, -torch.iinfo(qdtype).min]).min().to(weight.device, dtype=dtype) # TODO: check if doing scale/offset calculations on gpu is optimal # TODO: find cleaner way to get min
    elif qdtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return weight.type(qdtype).to(device), torch.tensor(1, dtype=dtype).to(device), torch.tensor(0, dtype=dtype).to(device)
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

    return remaining.round().type(qdtype).to(device), scale.to(device), offset.to(device)

def diag_tile_block(block, reps):
    assert block.shape[-1] == block.shape[-2]
    row = torch.nn.functional.pad(block, (0, block.shape[-1] * (reps - 1), 0, 0))
    return torch.concat(
        [torch.roll(row, block.shape[-1] * i, 1) for i in range(0, reps)]
    )

# def get_almost_hadamard(size: int):
#     tile_size = 1
#     while size % tile_size == 0:
#         tile_size *= 2
#     tile_size //= 2
#     tile_count = size // tile_size

#     temp = torch.tensor(hadamard(tile_size), dtype=torch.float32) / torch.sqrt(torch.tensor(tile_size))
#     m = diag_tile_block(temp, tile_count)

#     m_inv = m.T
#     m = m.type(dtype)
#     m_inv = m_inv.type(dtype)
#     return m, m_inv

num_test_hadamards = 1
num_orthogonality_tests = 100
max_allowed_inv_value = 1000
dot_threshold = 0.5
fail_print_interval = 10

cached_rotations = {}
def random_rotation_almost_hadamard(size: int, use_hardcoded, run_full_orthogonality_tests, check_inv_max):
    if use_hardcoded:
        tile_size = 1
        while size % tile_size == 0:
            tile_size *= 2
        tile_size //= 2
        tile_count = size // tile_size

        temp = torch.tensor(hadamard(tile_size), dtype=torch.float32) / torch.sqrt(torch.tensor(tile_size))
        m = diag_tile_block(temp, tile_count)

        m_inv = m.T
        m = m.type(torch.float16)
        m_inv = m_inv.type(torch.float16)
        return m, m_inv
    else:
        if size in cached_rotations:
            return cached_rotations[size]

        fail_count = 0
        potential = []
        while len(potential) < num_test_hadamards:
            try:
                m = torch.where(torch.rand((size, size)) >= 0.5, -1, 1).type(torch.float32) / torch.sqrt(torch.tensor(size))

                avg_row_dot_prod = 0
                tests_passed = True
                if run_full_orthogonality_tests:
                    for i in range(size):
                        for j in range(i + 1, size):
                            dot_prod = torch.abs(torch.nn.functional.cosine_similarity(m[i], m[j], dim=0))
                            if dot_prod > dot_threshold:
                                tests_passed = False
                                break
                            avg_row_dot_prod += dot_prod
                        if not tests_passed:
                            break
                else:
                    for _ in range(num_orthogonality_tests):
                        i, j = 0, 0
                        while i == j:
                            i, j = random.randrange(size), random.randrange(size)
                        dot_prod = torch.abs(torch.nn.functional.cosine_similarity(m[i], m[j], dim=0))
                        if dot_prod > dot_threshold:
                            tests_passed = False
                            break
                        avg_row_dot_prod += dot_prod
                
                if not tests_passed:
                    fail_count += 1
                    if fail_count % fail_print_interval == 0:
                        print(f"failed {fail_count} times")
                    continue
                avg_row_dot_prod /= (size - 1) * (size - 2)

                # since this isn't quite a hadamard matrix, it might have an extreme inverse
                # if it's too extreme, it could cause inf in float16 when multiplied; also
                # restricting maximum value in inverse could make the inverse closer to a
                # rotation matrix, which is ideal
                m_inv = torch.inverse(m).type(torch.float16)
                # TODO: determine what max value is acceptable
                if not check_inv_max or torch.max(torch.square(m_inv).sum(dim=1).sqrt()) < max_allowed_inv_value:
                    potential.append((m, avg_row_dot_prod))
                else:
                    fail_count += 1
                    if fail_count % fail_print_interval == 0:
                        print(f"failed {fail_count} times")
                
            except Exception as e:
                print(e)
                pass
        m, _ = min(potential, key=lambda x: x[1])

        m_inv = torch.inverse(m)
        m = m.type(torch.float16)
        m_inv = m_inv.type(torch.float16)

        cached_rotations[size] = (m, m_inv)
        
        return m, m_inv


ln_attn = ['error'] * 32
ln_ffn = ['error'] * 32

rots = []
sizes = [4096, 128, 128, 11008]
swap = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
for size in sizes:
    # tiled = diag_tile_block(swap, size // 2)
    # rots.append((tiled, tiled))
    # rots.append((torch.eye(size, dtype=dtype), torch.eye(size, dtype=dtype)))
    r, r_inv = random_rotation_almost_hadamard(size, use_hardcoded=False, run_full_orthogonality_tests=False, check_inv_max=True)
    rots.append((r, r_inv))
    # rots.append((r, r.T))

# idx = 1
# tiled = diag_tile_block(swap, sizes[idx] // 2)
# rots[idx] = (tiled, tiled)
rots[1] = (diag_tile_block(rots[1][0], 32), diag_tile_block(rots[1][1], 32))

def init(device):
    global rots
    for i in range(len(rots)):
        rots[i] = (rots[i][0].to(device), rots[i][1].to(device))

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

