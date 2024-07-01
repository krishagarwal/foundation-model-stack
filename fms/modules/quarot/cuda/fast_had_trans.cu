#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#ifndef __CUDACC__
#define __launch_bounds__(x,y)
#endif

#define MAX_WARPS_PER_SM 48

template <typename T, uint32_t THREADBLOCK_SIZE_M, uint32_t THREADBLOCK_SIZE_N>

// a_ptr, M: tl.constexpr, N: tl.constexpr, stride_m, stride_n, had_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
__global__ void __launch_bounds__(THREADBLOCK_SIZE_M * THREADBLOCK_SIZE_N, (MAX_WARPS_PER_SM * 32) / (THREADBLOCK_SIZE_M * THREADBLOCK_SIZE_N))
cuda_fast_had_trans(T* a, uint32_t M, uint32_t N, uint32_t stride_m, uint32_t stride_n, uint32_t had_size)
{
    __shared__ T chunk[THREADBLOCK_SIZE_N][THREADBLOCK_SIZE_M * 2];
    uint32_t idx_x = blockIdx.x * THREADBLOCK_SIZE_M + threadIdx.x;
    uint32_t idx_y = blockIdx.y * THREADBLOCK_SIZE_N + threadIdx.y;

    if (idx_y >= N) return;

    // chunk[threadIdx.x][threadIdx.y] = a[idx_x * stride_m + idx_y * stride_n];
    // chunk[threadIdx.x + THREADBLOCK_SIZE_M][threadIdx.y] = a[(idx_x + THREADBLOCK_SIZE_M) * stride_m + idx_y * stride_n];
    chunk[threadIdx.y][threadIdx.x] = a[idx_x * stride_m + idx_y * stride_n];
    chunk[threadIdx.y][threadIdx.x + THREADBLOCK_SIZE_M] = a[(idx_x + THREADBLOCK_SIZE_M) * stride_m + idx_y * stride_n];

    __syncthreads();

    for (uint32_t h = 1; h < had_size; h *= 2) {
        uint32_t my_id = (threadIdx.x % h) + (threadIdx.x / h) * h * 2;
        // my_id = (m_range % h) + (m_range // h) * h * 2
        half x = chunk[threadIdx.y][my_id];
        half y = chunk[threadIdx.y][my_id + h];
        chunk[threadIdx.y][my_id] = (x + y) * ((half)0.7071067811865475);
        chunk[threadIdx.y][my_id + h] = (x - y) * ((half)0.7071067811865475);
        if (h >= 32)
            __syncthreads();
    }

    a[idx_x * stride_m + idx_y * stride_n] = chunk[threadIdx.y][threadIdx.x];
    a[(idx_x + THREADBLOCK_SIZE_M) * stride_m + idx_y * stride_n] = chunk[threadIdx.y][threadIdx.x + THREADBLOCK_SIZE_M];
}


// template <typename T>
void fast_had_trans(uint64_t a, uint32_t M, uint32_t N, uint32_t stride_m, uint32_t stride_n, uint32_t had_size)
{
    constexpr uint32_t threadblock_size_n = 8;
    cuda_fast_had_trans<half, 64, threadblock_size_n><<<dim3(M / (64 * 2), (N + threadblock_size_n - 1) / threadblock_size_n), dim3(64, threadblock_size_n)>>>((half*)a, M, N, stride_m, stride_n, had_size);
    auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess)
        printf(cudaGetErrorString(status));
    // return 0;
    // printf("hello yeet");
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 yeet";

    m.def("fast_had_trans", &fast_had_trans, "test func");//, py::arg("a"), py::arg("had_size"));
}