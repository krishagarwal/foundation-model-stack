#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#include <cuda_fp16.h>
#include <cuda/annotated_ptr>

#ifndef __CUDACC__
#define __launch_bounds__(x,y)
#endif

#define MAX_WARPS_PER_SM 48

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// template <typename T, uint32_t THREADBLOCK_SIZE_M, uint32_t THREADBLOCK_SIZE_N>
template<int num_chunks, int warps_per_block>
// a_ptr, M: tl.constexpr, N: tl.constexpr, stride_m, stride_n, had_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
__global__ void __launch_bounds__(32 * warps_per_block, MIN(24, 48 / warps_per_block)) //__launch_bounds__(THREADBLOCK_SIZE_M * THREADBLOCK_SIZE_N, std::min(24u, (MAX_WARPS_PER_SM * 32) / (THREADBLOCK_SIZE_M * THREADBLOCK_SIZE_N))) // TODO: don't hardcode
// a is column major, b is row major
wmma_ker(half* __restrict__ a, half* __restrict__ hads, uint32_t stride_r, uint32_t stride_c, int log_had_size) {
    constexpr uint32_t chunk_stride_c = 16; // chunk will be column major
    __shared__ half chunk_1[16 * chunk_stride_c * warps_per_block]; // 16 * 32 chunk, col major (32x16 in mem)
    __shared__ half chunk_2[16 * chunk_stride_c * warps_per_block];
    half* __restrict__ chunk1 = chunk_1 + 16 * chunk_stride_c * (threadIdx.x / 32);
    half* __restrict__ chunk2 = chunk_2 + 16 * chunk_stride_c * (threadIdx.x / 32);
    uint blockid = blockIdx.x * warps_per_block + threadIdx.x / 32;
    uint threadid = threadIdx.x % 32;
    bool is_ch1 = true;
    a += blockid * stride_c * 16 * num_chunks; // swapped blockidx.x,y cause 64k limit
    cuda::access_property ap(cuda::access_property::streaming{});
    cuda::annotated_ptr<half, cuda::access_property> a_ann{a, ap}; 
    #pragma unroll
    for(int i = 0; i < 8; i++){
        uint32_t a_idx = i * stride_c * 2 + threadid * stride_r;
        half* a_loc = a + a_idx;
        uint16_t val;
        asm(
            "ld.global.L1::no_allocate.L2::256B.u16 %0, [%1];\n" : "=h"(val) : "l"(a_loc)
        ); // .L1::no_allocate .lu .L2::256B
        // (is_ch1 ? chunk1 : chunk2)[chunk_idx] = a_ann[a_idx];
        *(uint16_t*)(&(is_ch1 ? chunk1 : chunk2)[i * chunk_stride_c * 2 + threadid]) = val;
    }
    // __syncthreads();
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> had_frag[2];
    int remaining_log_had_size2 = log_had_size;
    // __shared__ half temp_had[16 * 16 * 2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int c_log_h = min(4, remaining_log_had_size2);
        wmma::load_matrix_sync(had_frag[i], &hads[256 * (c_log_h - 1)], 16);
        remaining_log_had_size2 -= 4;
        if(remaining_log_had_size2 <= 0) break;
    }

    #pragma unroll
    for (int k = 0; k < num_chunks; k++) {
        if (k < num_chunks - 1) {
            #pragma unroll
            for(int j = 0; j < 16; j += 2){
                uint32_t a_idx = j * stride_c + threadid * stride_r + 256 * (k + 1);
                half* a_loc = a + a_idx;
                uint16_t val;
                asm(
                    "ld.global.L1::no_allocate.L2::256B.u16 %0, [%1];\n" : "=h"(val) : "l"(a_loc)
                );
                *(uint16_t*)(&(!is_ch1 ? chunk1 : chunk2)[j * chunk_stride_c + threadid]) = val;
                // (!is_ch1 ? chunk1 : chunk2)[j * chunk_stride_c + threadid] = a_ann[j * stride_c + threadid * stride_r + 256 * (k + 1)]; // + 256 * (k + 1) assumes a is column-major
            }
        }

        int remaining_log_had_size = log_had_size;

        #pragma unroll
        for(int i = 0; i < 2; i++) {
            // int c_log_h = min(4, remaining_log_had_size);

            // Define the fragments
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
            // Initialize the output to zero
            wmma::fill_fragment(c_frag, 0.0f);
            // Load the inputs
            wmma::load_matrix_sync(b_frag, (is_ch1 ? chunk1 : chunk2), chunk_stride_c);
            // Perform the matrix multiplication
            wmma::mma_sync(c_frag, had_frag[i], b_frag, c_frag);

            remaining_log_had_size -= 4;
            wmma::layout_t mode = (remaining_log_had_size <= 0 && i == 0) ? wmma::mem_col_major : wmma::mem_row_major;
            wmma::store_matrix_sync((is_ch1 ? chunk1 : chunk2), c_frag, chunk_stride_c, mode); //col_major); // we store row major to get a free transpose

            if(remaining_log_had_size <= 0) break;
        }

        #pragma unroll
        for(int j = 0; j < 16; j += 2){
            a_ann[j * stride_c + threadid * stride_r + 256 * k] = (is_ch1 ? chunk1 : chunk2)[j * chunk_stride_c + threadid];
        }

        // half* temp = chunk;
        // chunk = next_chunk;
        // next_chunk = temp;
        is_ch1 = !is_ch1;
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
    uint32_t vector_size = 256;
    uint32_t log_had_size = 8;
    uint32_t cols = 256;//4096 * 16;
    constexpr int chunks_per_warp = 1;
    constexpr int warps_per_block = 1;

    half* ptr = (half*)malloc(vector_size * cols * sizeof(half)); // col major

    for(int i = 0; i < vector_size * cols; i++) ptr[i] = i % 2;//(i % cols == i / cols);//1;
    half* dev_ptr;
    cudaMalloc(&dev_ptr, vector_size * cols * sizeof(half));
    cudaMemcpy(dev_ptr, ptr, vector_size * cols * sizeof(half), cudaMemcpyKind::cudaMemcpyHostToDevice);

    half* dev_had_16;
    half had_16[4][16][16];
    cudaMalloc(&dev_had_16, sizeof(had_16));
    for(int s = 0; s < 4; s++)
        for(int r = 0; r < 16; r++)
            for(int c = 0; c < 16; c++)
                had_16[s][r][c] = r == c;

    half rsqrt2 = 1;//(half)(1/sqrt(2));

    for(int s = 0; s < 4; s++){
        int h = 1;
        while (h < (1 << (s + 1))){
            // perform FWHT
            // for i in range(0, len(a), h * 2):
            for (int i = 0; i < 16; i += h * 2){
                // for j in range(i, i + h):
                for (int j = i; j < i + h; j++){
                    for (int k = 0; k < 16; k++){
                        half x = had_16[s][j][k];
                        half y = had_16[s][j + h][k];
                        had_16[s][j][k] = (x + y) * rsqrt2;
                        had_16[s][j + h][k] = (x - y) * rsqrt2;
                    }
                }
            }
            h *= 2;
        }
    }

    cudaMemcpy(dev_had_16, had_16, sizeof(had_16), cudaMemcpyKind::cudaMemcpyHostToDevice);
    printf("cols: %d\n", cols);
    wmma_ker<chunks_per_warp, warps_per_block><<<dim3(((cols / 16) * (vector_size / 16)) / chunks_per_warp / warps_per_block, 1), dim3(32 * warps_per_block)>>>(dev_ptr, dev_had_16, 1, 16, log_had_size);
    auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        printf(cudaGetErrorString(status));
        return 1;
    }


    cudaMemcpy(ptr, dev_ptr, vector_size * cols * sizeof(half), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    // for(int j = 0; j < cols; j++){
    //     if (ptr[0 * 1 + j * vector_size] != (half)sqrt(1 << log_had_size)) {
    //         printf("error: %f", (float)ptr[0 * 1 + j * vector_size]);
    //         return 1;
    //     }
    // }

    printf("result: [\n");
    // col major
    for(int i = 0; i < min(vector_size, 32); i++) {
        printf("    [");
        constexpr int start_show_col = 0;
        for(int j = start_show_col; j < min(cols, start_show_col + 32); j++) {
            printf("%.1f ", (float)ptr[i * 1 + j * vector_size]);
        }
        printf("]\n");
    }
    printf("]\n");

    printf("bye\n");
    return 0;
}