#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#include <cuda_fp16.h>

#ifndef __CUDACC__
#define __launch_bounds__(x,y)
#endif

#define MAX_WARPS_PER_SM 48

// template <typename T, uint32_t THREADBLOCK_SIZE_M, uint32_t THREADBLOCK_SIZE_N>

// a_ptr, M: tl.constexpr, N: tl.constexpr, stride_m, stride_n, had_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
__global__ void //__launch_bounds__(THREADBLOCK_SIZE_M * THREADBLOCK_SIZE_N, std::min(24u, (MAX_WARPS_PER_SM * 32) / (THREADBLOCK_SIZE_M * THREADBLOCK_SIZE_N))) // TODO: don't hardcode
// a is column major, b is row major
wmma_ker(half *a, half *had_16, uint32_t stride_r, uint32_t stride_c) {
    __shared__ half chunk[32 * 16];// 16 * 32 chunk, col major (32x16 in mem)
    a += blockIdx.y * stride_r * 16 + blockIdx.x * stride_c * 16 * 2; // swapped blockidx.x,y cause 64k limit
    #pragma unroll
    for(int i = 0; i < 16; i++){
        chunk[i * stride_c * 2 + threadIdx.x * stride_r] = a[i * stride_c * 2 + threadIdx.x * stride_r];
    }
    // __syncthreads();
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag_1;
    wmma::load_matrix_sync(a_frag_1, had_16, 16);

    #pragma unroll
    for(int i = 0; i < 2; i++) {

        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag_1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag_1;
        // Declare the fragments
        // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag_2;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag_2;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag_2;

        // Initialize the output to zero
        wmma::fill_fragment(c_frag_1, 0.0f);
        // Initialize the output to zero
        wmma::fill_fragment(c_frag_2, 0.0f);

        // Load the inputs
        wmma::load_matrix_sync(b_frag_1, chunk, stride_c);

        // Load the inputs
        // wmma::load_matrix_sync(a_frag_2, had_16, 16);
        wmma::load_matrix_sync(b_frag_2, chunk + stride_c * 16, stride_c);

        // Perform the matrix multiplication
        wmma::mma_sync(c_frag_1, a_frag_1, b_frag_1, c_frag_1);
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag_2, a_frag_1, b_frag_2, c_frag_2);

        // Store the output
        wmma::store_matrix_sync(chunk, c_frag_1, stride_c, wmma::mem_row_major); //col_major); // we store row major to get a free transpose
        // Store the output
        wmma::store_matrix_sync(chunk + stride_c * 16, c_frag_2, stride_c, wmma::mem_row_major); //col_major); // we store row major to get a free transpose
    }
    //shuffle

    // def part_trans(a, h, all_h_sizes, index_in_all):
    // a_trans = h @ a.view(*all_h_sizes, -1).transpose(0, index_in_all)
    
    // (16, 16, 2)
    // (16, 16, 2)
    // (1, 0, 2)



    // do 8-16x factor again

    // shuffle back to return
    // return a_trans.transpose(0, index_in_all).reshape_as(a) # (-1, 1)

    // __syncthreads();
    #pragma unroll
    for(int i = 0; i < 16; i++){
        a[i * stride_c * 2 + threadIdx.x * stride_r] = chunk[i * stride_c * 2 + threadIdx.x * stride_r];
    }
}


// template <typename T>
int main()
{
    // py::print("hi", true);
    printf("hi\n");
    // constexpr uint32_t threadblock_size_n = 8;
    // constexpr uint32_t threadblock_size_m = 64;

    // 16x4096
    uint32_t had_size = 256;
    uint32_t cols = 4096 * 16;

    half* ptr = (half*)malloc(had_size * cols * sizeof(half)); // col major

    for(int i = 0; i < had_size * cols; i++) ptr[i] = 1;
    half* dev_ptr;
    cudaMalloc(&dev_ptr, had_size * cols * sizeof(half));
    cudaMemcpy(dev_ptr, ptr, had_size * cols * sizeof(half), cudaMemcpyKind::cudaMemcpyHostToDevice);

    half* dev_had_16;
    cudaMalloc(&dev_had_16, 16 * 16 * sizeof(half));
    half had_16[16][16];
    for(int r = 0; r < 16; r++)
        for(int c = 0; c < 16; c++)
            had_16[r][c] = r == c;

    half rsqrt2 = (half)(1/sqrt(2));

    int h = 1;
    while (h < 16) {
        // perform FWHT
        // for i in range(0, len(a), h * 2):
        for (int i = 0; i < 16; i += h * 2){
            // for j in range(i, i + h):
            for (int j = i; j < i + h; j++){
                for (int k = 0; k < 16; k++){
                    half x = had_16[j][k];
                    half y = had_16[j + h][k];
                    had_16[j][k] = (x + y) * rsqrt2;
                    had_16[j + h][k] = (x - y) * rsqrt2;
                }
            }
        }
        // normalize and increment
        // a /= math.sqrt(2)
        h *= 2;
    }

    cudaMemcpy(dev_had_16, had_16, sizeof(had_16), cudaMemcpyKind::cudaMemcpyHostToDevice);
    printf("cols: %d\n", cols);
    wmma_ker<<<dim3(cols / 16 / 2 * had_size / 16, 1), dim3(32)>>>(dev_ptr, dev_had_16, 1, 16);
    auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        printf(cudaGetErrorString(status));
        return 1;
    }


    cudaMemcpy(ptr, dev_ptr, had_size * cols * sizeof(half), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    for(int j = 0; j < cols; j++){
        if (ptr[0 * 1 + j * had_size] != (half)16.0f) {
            printf("error: %f", (float)ptr[0 * 1 + j * had_size]);
            return 1;
        }
    }

    printf("result: [\n");
    // col major
    for(int i = 0; i < min(had_size, 16); i++) {
        printf("    [");
        constexpr int start_show_col = 0;
        for(int j = start_show_col; j < min(cols, start_show_col + 32); j++) {
            printf("%.1f ", (float)ptr[i * 1 + j * had_size]);
        }
        printf("]\n");
    }
    printf("]\n");

    printf("bye\n");
    return 0;
}