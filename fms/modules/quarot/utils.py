import torch
from scipy.linalg import hadamard
import random
from kmeans_gpu import KMeans
from .fast_had_trans import triton_fast_had

dtype    = torch.float16
qdtype   = torch.float16 # 8_e4m3fn # 16 # int8  #_e5m2
accdtype = torch.float16 # int32
use_quant_map = False
skip_bad_layers = False
test_against_truth = False
test_float_range = (1, 100) #(0.01, 6)# None
test_float_vals = 1
current_float_val = test_float_range[0]
current_score = 0
use_hadamard = True

temp_layer = 0

def quantize(weight: torch.Tensor, qdtype, dim=-1, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = weight.device
    
    # TODO: make sure it works for all dtypes
    if qdtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        # temp, _ = weight.abs().max(dim=dim, keepdim=True)
        # temp = 1#2.4
        # temp = current_float_val
        # temp = torch.tensor(temp)
        temp = torch.sqrt(torch.tensor(65504 * current_float_val / weight.shape[dim]))
        scale_max = temp.to(weight.device, dtype=dtype)
        # return weight.type(qdtype), torch.tensor(1, dtype=dtype).to(device)
        # scale_max = torch.tensor([torch.finfo(qdtype).max, -torch.finfo(qdtype).min]).min().to(weight.device, dtype=dtype) # TODO: check if doing scale/offset calculations on gpu is optimal # TODO: find cleaner way to get min
    elif qdtype in [torch.int8, torch.int16]:
        scale_max = torch.tensor([torch.iinfo(qdtype).max, -torch.iinfo(qdtype).min]).min().to(weight.device, dtype=dtype) # TODO: check if doing scale/offset calculations on gpu is optimal # TODO: find cleaner way to get min
    elif qdtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return weight.type(qdtype).to(device), torch.tensor(1, dtype=dtype).to(device)
    else:
        raise ValueError("type not supported :(")

    mag, _ = weight.abs().max(dim=dim, keepdim=True)
    scale = mag / scale_max
    weight = weight / scale
    if qdtype in [torch.int8, torch.int16]:
        weight = weight.round()
    return weight.type(qdtype).to(device), scale.to(device)

def diag_tile_block(block, reps):
    assert block.shape[-1] == block.shape[-2]
    row = torch.nn.functional.pad(block, (0, block.shape[-1] * (reps - 1), 0, 0))
    return torch.concat(
        [torch.roll(row, block.shape[-1] * i, 1) for i in range(0, reps)]
    )

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

        temp = torch.tensor(hadamard(tile_size), dtype=torch.float32) * torch.rsqrt(torch.tensor(tile_size))
        m = diag_tile_block(temp, tile_count)

        m_inv = m.T
        m = m.type(dtype)
        m_inv = m_inv.type(dtype)
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
        m = m.type(dtype)
        m_inv = m_inv.type(dtype)

        cached_rotations[size] = (m, m_inv)
        
        return m, m_inv


ln_attn = ['error'] * 32
ln_ffn = ['error'] * 32
dec_norm_weight = 'error'

rots = []
sizes = [4096, 128, 128, 11008]
for size in sizes:
    if not use_hadamard:
        rots.append((torch.eye(size, dtype=dtype), torch.eye(size, dtype=dtype)))
    else:
        r, r_inv = random_rotation_almost_hadamard(size, use_hardcoded=True, run_full_orthogonality_tests=False, check_inv_max=True)
        rots.append((r, r_inv))
    # rots.append((r, r.T))

# idx = 1
# swap = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
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

# def get_pre_rot(key_steps):
#     if weight_check(key_steps, ['wg', 'w1']):
#         return rots[0][1] @ torch.diag(ln_ffn[int(key_steps[1])].type(dtype)) #.view(1, -1).type(dtype) # [2] is the block number
#     if weight_check(key_steps, 'w2'):
#         return rots[3][1]
#     if weight_check(key_steps, ['query', 'key', 'value']):
#         return rots[0][1] @ torch.diag(ln_attn[int(key_steps[1])].type(dtype)) #.view(1, -1).type(dtype) # will brodcast across columns, equivalent of a diagonal matrix
#     if weight_check(key_steps, 'dense'):
#         return rots[1][1]
#     else:
#         return None

# def get_post_rot(key_steps):
#     if weight_check(key_steps, 'dense'):
#         return rots[0][0]
#     if weight_check(key_steps, 'value'):
#         return rots[1][0]
#     if weight_check(key_steps, 'w2'):
#         return rots[0][0]
#     else:
#         return None

def right_had(a, had_size=None):
    return triton_fast_had(a.transpose(-2, -1), had_size=had_size).transpose(-2, -1)

def apply_pre_rot(key_steps, a):
    if weight_check(key_steps, ['wg', 'w1']):
        a = torch.diag(ln_ffn[int(key_steps[1])].type(dtype)) @ a # [2] is the block number
    elif weight_check(key_steps, ['query', 'key', 'value']):
        a = torch.diag(ln_attn[int(key_steps[1])].type(dtype)) @ a # will brodcast across columns, equivalent of a diagonal matrix
    elif weight_check(key_steps, 'head'):
        a = torch.diag(dec_norm_weight).type(dtype) @ a

    if not use_hadamard:
        return a
    if weight_check(key_steps, ['wg', 'w1', 'head']):
        return triton_fast_had(a) # rots[0][1] @ a
    if weight_check(key_steps, 'w2'):
        return triton_fast_had(a, had_size=256) # rots[3][1] @ a # TODO: don't hardcode
    if weight_check(key_steps, ['query', 'key', 'value']):
        return triton_fast_had(a) # rots[0][1] @ a
    if weight_check(key_steps, 'dense'):
        return triton_fast_had(a, had_size=128) # rots[1][1] @ a # TODO: don't hardcode
    return a

def apply_post_rot(key_steps, a):
    if not use_hadamard:
        return a
    if weight_check(key_steps, ['dense', 'emb']):
        return right_had(a) # a @ rots[0][0]
    if weight_check(key_steps, 'value'):
        return right_had(a, had_size=128) # a @ rots[1][0] # TODO: don't hardcode
    if weight_check(key_steps, 'w2'):
        return right_had(a) # a @ rots[0][0]
    return a

def quantize_cluster(weights: torch.Tensor):
    temp_device = weights.device
    centroids = KMeans(n_clusters=256)(weights.view(1, -1, 1).to(torch.float32))
    centroids = centroids.to(torch.float16)
    # avgs = [x[0] for x in kmeans.cluster_centers_]
    dist = weights.view(weights.shape[0], weights.shape[1], 1) - centroids.view(1, 1, 16)
    abs_dist = torch.abs(dist)
    closest_idx = torch.argmin(abs_dist, dim=2).to(torch.uint8) # TODO: choose index dtype
    return closest_idx.to(temp_device), centroids.view(-1)

def dequantize_cluster(weights: torch.Tensor, avgs: torch.Tensor):
    final_result = torch.zeros_like(weights)
    for i, val in enumerate(avgs):
        final_result = torch.where(weights == i, val, final_result)
    return final_result
