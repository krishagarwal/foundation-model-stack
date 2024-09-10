from fast_hadamard_transform import hadamard_transform
import torch
import scipy.linalg
import math
import tensor_core_had_async

# hadamard sizes
test_sizes_m = [2, 4, 8, 16, 32, 64, 128, 256, 4096, 8192, 16384, 32768]#, 512, 1024, 2048, 4096, 8192, 16384, 32768]
# batch sizes
test_sizes_n = [1 << i for i in range(25)] # 32M max
max_elements = 64 * (1 << 20)
runs_per_size = 10

test_count = len(test_sizes_m) * len(test_sizes_n) * runs_per_size
tests_done = 0
failed_tests = 0

def get_scale(size):
    return math.sqrt(1 / size)

truth_hadamards = [(torch.tensor(scipy.linalg.hadamard(size), device='cuda', dtype=torch.float32) * get_scale(size)).to(torch.float16) for size in test_sizes_m]

def truth_hadamard_transform(a: torch.Tensor):
    # target_index = -1
    # for i in range(len(test_sizes_m)):
    #     if test_sizes_m[i] == a.shape[1]:
    #         target_index = i
    #         break
    # return (a @ truth_hadamards[int(target_index)]).T
    return hadamard_transform(a.clone(), get_scale(a.shape[1])).T

def test_hadamard_transform(a: torch.Tensor):
    return hadamard_transform(a, get_scale(a.shape[1])).T
    # a_clone = torch.empty(a.shape, device='cuda', dtype=torch.float16)
    # a_clone.copy_(a)
    # tensor_core_had_async.fast_had_trans_async(a_clone.data_ptr(), a.shape[0], a.shape[1], int(math.log2(a.shape[1])))
    # return a_clone.T

torch.manual_seed(0)

for m in test_sizes_m:
    print(f'Testing size {m}xN')
    for n in test_sizes_n:
        if m * n > max_elements:
            tests_done += runs_per_size
            if tests_done % 100 == 0 or tests_done == test_count:
                print(f'{tests_done}/{test_count} tests done')
            continue

        a = torch.randn((m, n), device='cuda', dtype=torch.float16)
        # a = torch.zeros((m, n), device='cuda', dtype=torch.float16)
        # for i in range(min(a.shape[0], a.shape[1])):
        #     a[i, i] = 1.0
        for i in range(runs_per_size):
            # run test here
            truth = truth_hadamard_transform(a.T)
            result = test_hadamard_transform(a.T)

            success = torch.allclose(truth, result, atol=1e-2, rtol=0) # TODO: NOTE: we are not accurate down to 3 decimal places(atol)

            if not success:
                torch.set_printoptions(threshold=100)
                print(f'Failed test: {m}x{n}')
                print(f'Input:')
                print(a)
                print(f'Expected:')
                print(truth)
                print(f'Got:')
                print(result)
                # worst element
                diff = torch.abs(truth - result)
                max_diff = torch.max(diff)
                print(f'Max diff: {max_diff}')
                print(f'Max diff index: {torch.argmax(diff)}')
                diff_input = torch.abs(a - result)
                max_diff_input = torch.max(diff_input)
                print(f'Max diff input: {max_diff_input}')
                print('')
                exit(1)
                # failed_tests += 1
                # if failed_tests >= 20:
                #     print('Too many failed tests, aborting')
                #     exit(1)

            tests_done += 1
            if tests_done % 100 == 0 or tests_done == test_count:
                print(f'{tests_done}/{test_count} tests done')