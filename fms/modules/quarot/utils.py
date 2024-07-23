import torch
from scipy.linalg import hadamard
import random
# from kmeans_gpu import KMeans
from .fast_had_trans import left_had, right_had
from .special_had import get_had172

dtype          = torch.float16
qdtype         = torch.int8 # float16 # 8_e4m3fn # 16 # int8  #_e5m2
uqdtype        = torch.uint8
accdtype       = torch.int32 # float16 # float16 # int32
scaledtype     = torch.float16
bits = 8
# max_qint = torch.tensor(127) # 0 if qdtype not in [torch.int8, torch.int16, torch.int32] else torch.tensor([torch.iinfo(qdtype).max, -torch.iinfo(qdtype).min]).min()

use_quant_map = False
skip_bad_layers = False
test_against_truth = False
test_float_range = (1, 100) #(0.01, 6)# None
test_float_vals = 1
current_float_val = test_float_range[0]
current_score = 0
use_hadamard = True
use_graph = False # TODO: this is broken

temp_layer = 0

weight_clip_ratio = 1
activ_clip_ratio = 0.9
kv_cache_clip_ratio = 0.95

def quantize(weight: torch.Tensor, qdtype, dim=-1, device=None, sym=True, use_mse=False, clip_ratio=1) -> tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = weight.device
    
    # TODO: make sure it works for all dtypes
    if qdtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
        raise Exception("this doesn't work anymore :(")
        # temp, _ = weight.abs().max(dim=dim, keepdim=True)
        # temp = 1#2.4
        # temp = current_float_val
        # temp = torch.tensor(temp)
        temp = torch.sqrt(torch.tensor(65504 * current_float_val / weight.shape[dim]))
        scale_max = temp.to(weight.device, dtype=scaledtype)
        # return weight.type(qdtype), torch.tensor(1, dtype=dtype).to(device)
        # scale_max = torch.tensor([torch.finfo(qdtype).max, -torch.finfo(qdtype).min]).min().to(weight.device, dtype=dtype) # TODO: check if doing scale/offset calculations on gpu is optimal # TODO: find cleaner way to get min
    elif qdtype in [torch.int8, torch.int16]:
        max_qint = 2 ** (bits - 1) - 1 if sym else 2 ** bits - 1
        min_qint = -(max_qint + 1) if sym else 0
    elif qdtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        return weight.type(qdtype).to(device), torch.tensor(1, dtype=scaledtype).to(device), torch.tensor(0, dtype=scaledtype).to(device)
    else:
        raise ValueError("type not supported :(")

    if sym:
        offset = torch.tensor(0, dtype=qdtype, device=device)
        mag, _ = weight.abs().max(dim=dim, keepdim=True)
        mag *= clip_ratio
    else:
        weight_max = weight.max(dim=dim, keepdim=True)[0].maximum(torch.tensor(0, dtype=weight.dtype)) * clip_ratio
        weight_min = weight.min(dim=dim, keepdim=True)[0].minimum(torch.tensor(0, dtype=weight.dtype)) * clip_ratio
        is_zero = (weight_max == 0) & (weight_min == 0)
        weight_max[is_zero] = 1
        weight_min[is_zero] = -1
        mag = weight_max - weight_min
        offset = torch.round(-weight_min * max_qint / mag).to(scaledtype)
    scale = (mag / max_qint).to(scaledtype)

    # Adapted to match QuaRot
    if use_mse:
        err_shape = list(weight.shape)
        err_shape[dim] = 1 # weight shape except for dim, since only 1 scale along dim
        best_err = torch.full(err_shape, torch.inf, device=device)
        steps = 100
        min_frac = 0.2
        for i in range(int((1-min_frac) * steps)): # 0.2 * 100: min % to shrink to, how many steps to cut max into
            mag1 = (1 - i / steps) * mag

            scale1 = mag1 / max_qint # has right shape based on dim
            weight1 = weight / scale + offset
            if qdtype in [torch.int8, torch.int16]:
                weight1 = weight1.clamp(min=min_qint, max=max_qint).round()
            if sym:
                weight1 = weight1.type(qdtype)
            else:
                weight1 = weight1.type(uqdtype)
            weight1_comp = (scale1 * (weight1 - offset)).to(dtype)

            diff = weight - weight1_comp
            err = diff.abs().pow(2.4).sum(dim)

            # update best scale and err, per row
            is_new_best = err < best_err
            best_err = torch.where(is_new_best, err, best_err)
            scale = torch.where(is_new_best, scale1, scale)

    weight = weight / scale + offset
    if qdtype in [torch.int8, torch.int16]:
        weight = weight.clamp(min=min_qint, max=max_qint).round()
    if sym:
        weight = weight.type(qdtype)
    else:
        weight = weight.type(uqdtype)
    return weight.to(device), scale.to(device), offset.to(device)

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
def random_rotation_almost_hadamard(size: int, use_hardcoded, run_full_orthogonality_tests, check_inv_max, tile_size=None):
    if use_hardcoded:
        if tile_size is None:
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
tile_sizes = [None, None, None, 64]
for size, tile_size in zip(sizes, tile_sizes):
    if not use_hadamard:
        rots.append((torch.eye(size, dtype=dtype), torch.eye(size, dtype=dtype)))
    else:
        r, r_inv = random_rotation_almost_hadamard(size, use_hardcoded=True, run_full_orthogonality_tests=False, check_inv_max=True, tile_size=tile_size)
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
    global had172
    had172 = get_had172(device) / torch.sqrt(torch.tensor(172, dtype=torch.float16, device=device))
    global rand_diag
    torch.manual_seed(0)
    rand_diag = torch.diag(torch.randint(1, 2, (4096,), device=device, dtype=torch.float16) * 2 - 1)

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
        return left_had(rand_diag @ a) # rots[0][1] @ a
    if weight_check(key_steps, 'w2'):
        rot64 = left_had(a, had_size=64)
        out = torch.zeros(rot64.shape[::-1], dtype=rot64.dtype, device=rot64.device)
        out.copy_(rot64.T)
        out = (out.reshape(-1, 172, 64).transpose(-1, -2) @ had172).transpose(-1, -2).reshape_as(out)
        rot64.copy_(out.T)
        return rot64
        # return (had172 @ rot64.view(64, 172, -1).transpose(0, 1).reshape(172, -1)).view(172, 64, -1).transpose(0, 1).reshape(11008, -1)
        # return triton_fast_had(a, had_size=256) # rots[3][1] @ a # TODO: don't hardcode
    if weight_check(key_steps, ['query', 'key', 'value']):
        return left_had(rand_diag @ a) # rots[0][1] @ a
    if weight_check(key_steps, 'dense'):
        return left_had(a) # rots[1][1] @ a # TODO: don't hardcode # TODO: bring back rand_diag here
    return a

def apply_post_rot(key_steps, a):
    if not use_hadamard:
        return a
    if weight_check(key_steps, ['dense', 'emb']):
        return right_had(a @ rand_diag) # a @ rots[0][0]
    if weight_check(key_steps, 'value'):
        return right_had(a, had_size=128) # a @ rots[1][0] # TODO: don't hardcode # TODO: bring back rand_diag here
    if weight_check(key_steps, 'w2'):
        return right_had(a @ rand_diag) # a @ rots[0][0]
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
