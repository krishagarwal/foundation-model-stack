from fast_hadamard_transform import hadamard_transform
import torch
import scipy.linalg
import math
import tensor_core_had_async

# set to false to check performance
correctness_check = False
# set to warmup count + 1 to check performance
runs_per_size = 5

# hadamard sizes
test_sizes_m = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768] # 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 #2, 4, 8, 16, 32, 64, 

test_elem_counts = [1 << i for i in range(9, 26, 1)] # 32MB # 64MB # 2**28 = 256M

print("test_sizes_m: ", test_sizes_m)
print("test_elem_counts: ", test_elem_counts)

test_count = len(test_sizes_m) * len(test_elem_counts)
tests_done = 0
failed_tests = 0

def get_scale(size):
    return math.sqrt(1 / size)

# truth_hadamards = [(torch.tensor(scipy.linalg.hadamard(size), device='cuda', dtype=torch.float32) * get_scale(size)).to(torch.float16) for size in test_sizes_m]

def truth_hadamard_transform_inplace(a: torch.Tensor):
    # target_index = -1
    # for i in range(len(test_sizes_m)):
    #     if test_sizes_m[i] == a.shape[1]:
    #         target_index = i
    #         break
    # return (a @ truth_hadamards[int(target_index)]).T
    return hadamard_transform(a, get_scale(a.shape[1]))

def test_hadamard_transform_inplace_rowmajor(a: torch.Tensor):
    # return hadamard_transform(a, get_scale(a.shape[1])).T
    tensor_core_had_async.fast_had_trans_async(a.data_ptr(), a.shape[0], a.shape[1], int(math.log2(a.shape[1])))
    return a

torch.manual_seed(0)

def check_correctness(m, elem_c, a, result, truth):
    success = torch.allclose(truth, result, atol=1e-2, rtol=0) # TODO: NOTE: we are not accurate down to 3 decimal places(atol)

    if not success:
        torch.set_printoptions(threshold=100)
        print(f'Failed test: {m}x{elem_c // m}')
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

for m in test_sizes_m:
    for elem_c in test_elem_counts:
        if elem_c < m:
            tests_done += runs_per_size
            if tests_done % 100 == 0 or tests_done == test_count:
                print(f'{tests_done}/{test_count} tests done')
            continue
        print(f'Testing size {m}x{elem_c // m}')

        a = torch.randn((elem_c // m, m), device='cuda', dtype=torch.float16)
        # a = torch.zeros((m, elem_c // m), device='cuda', dtype=torch.float16)
        # for i in range(min(a.shape[0], a.shape[1])):
        #     a[i, i] = 1.0
        if correctness_check:
            for i in range(runs_per_size):
                # run test here
                a_result = a.clone()
                a_truth = a.clone()
                result = test_hadamard_transform_inplace_rowmajor(a_result)
                truth = truth_hadamard_transform_inplace(a_truth)

                check_correctness(m, elem_c, a, result, truth)
        else:
            # run in a row so that warmup is valid
            a_result = a # we can clobber the result cause we are only interested in timing
            for i in range(runs_per_size):
                a_result = test_hadamard_transform_inplace_rowmajor(a_result)
            a_truth = a
            for i in range(runs_per_size):
                a_truth = truth_hadamard_transform_inplace(a_truth)
            a_memcpy = a
            # also can compare timing to memcpy
            temp = torch.empty_like(a)
            for i in range(runs_per_size):
                temp.copy_(a_memcpy)
            # do nothing with results since we are only interested in timing
            # NOTE: make sure to disable clearing cache in Nsight Compute

        tests_done += 1
        if tests_done % 100 == 0 or tests_done == test_count:
            print(f'{tests_done}/{test_count} size tests done')