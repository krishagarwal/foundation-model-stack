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

// template<size_t n>
// struct FragHalf2 {
//     half2 data[n];

//     half2 operator[] (size_t i) {
//         return data[i];
//     }
// };
typedef uint32_t b32;

// a 4x2, b 2x2, c 2x2
// __device__ __forceinline__ void mma_m16_n8_k16_fp16_fp16_fp16_noacc(FragHalf2<4>& a, FragHalf2<2>& b, FragHalf2<2>& c) {
//     //d, a, b, c
//     asm (
//         "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
//         "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {RZ, RZ};\n\t"
//         : "=r"(c[0]), "=r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
//     );
// }
// // a 4x2, b 4x2, c 4x2
// __device__ __forceinline__ void mma_m16_n16_k16_fp16_fp16_fp16_noacc(FragHalf2<4>& a, FragHalf2<4>& b, FragHalf2<4>& c) {
//     mma_m16_n8_k16_fp16_fp16_fp16_noacc(a, *(FragHalf2<2>*)(&b[0]), *(FragHalf2<2>*)(&c[0]));
//     mma_m16_n8_k16_fp16_fp16_fp16_noacc(a, *(FragHalf2<2>*)(&b[2]), *(FragHalf2<2>*)(&c[2]));
// }

// a 4x2, b 2x2, c 2x2
__device__ __forceinline__ void mma_m16_n8_k16_fp16_fp16_fp16_noacc(b32 a0, b32 a1, b32 a2, b32 a3, b32 b0, b32 b1, b32& c0, b32& c1){
    //d, a, b, c
    b32 zero = 0;
    asm (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n\t"
        : "=r"(c0), "=r"(c1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(zero), "r"(zero)
    );
}
// a 4x2, b 4x2, c 4x2
__device__ __forceinline__ void mma_m16_n16_k16_fp16_fp16_fp16_noacc(b32 a0, b32 a1, b32 a2, b32 a3, b32 b0, b32 b1, b32 b2, b32 b3, b32& c0, b32& c1, b32& c2, b32& c3){
    mma_m16_n8_k16_fp16_fp16_fp16_noacc(a0, a1, a2, a3, b0, b1, c0, c1);
    mma_m16_n8_k16_fp16_fp16_fp16_noacc(a0, a1, a2, a3, b2, b3, c2, c3);
}

__device__ __forceinline__ void matrix_transpose_m8_n8_fp16_inplace(b32& a0) {
    asm (
        "movmatrix.sync.aligned.m8n8.trans.b16 "
        "%0, %1;\n\t"
        : "=r"(a0) : "r"(a0)
    );
}

// template <typename T, uint32_t THREADBLOCK_SIZE_M, uint32_t THREADBLOCK_SIZE_N>
template<int num_chunks, int warps_per_block>
// a_ptr, M: tl.constexpr, N: tl.constexpr, stride_m, stride_n, had_size: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
__global__ void __launch_bounds__(32 * warps_per_block, MIN(24, 48 / warps_per_block)) //__launch_bounds__(THREADBLOCK_SIZE_M * THREADBLOCK_SIZE_N, std::min(24u, (MAX_WARPS_PER_SM * 32) / (THREADBLOCK_SIZE_M * THREADBLOCK_SIZE_N))) // TODO: don't hardcode
// a is column major, b is row major
wmma_ker(half* __restrict__ a, half* __restrict__ hads, uint32_t stride_r, uint32_t stride_c, int log_had_size) {
    // constexpr uint32_t chunk_stride_c = 16; // chunk will be column major
    // __shared__ half chunk_1[16 * chunk_stride_c * warps_per_block]; // 16 * 32 chunk, col major (32x16 in mem)
    // __shared__ half chunk_2[16 * chunk_stride_c * warps_per_block];
    // half* __restrict__ chunk1 = chunk_1 + 16 * chunk_stride_c * (threadIdx.x / 32);
    // half* __restrict__ chunk2 = chunk_2 + 16 * chunk_stride_c * (threadIdx.x / 32);
    uint blockid = blockIdx.x * warps_per_block + threadIdx.x / 32;
    uint threadid = threadIdx.x % 32;
    // bool is_ch1 = true;
    a += blockid * stride_c * 16 * num_chunks; // swapped blockidx.x,y cause 64k limit
    
    // #pragma unroll
    // for(int i = 0; i < 8; i++){
    //     uint32_t a_idx = i * stride_c * 2 + threadid * stride_r;
    //     half* a_loc = a + a_idx;
    //     uint16_t val;
    //     asm(
    //         "ld.global.L1::no_allocate.L2::256B.u16 %0, [%1];\n" : "=h"(val) : "l"(a_loc)
    //     ); // .L1::no_allocate .lu .L2::256B
    //     // (is_ch1 ? chunk1 : chunk2)[chunk_idx] = a_ann[a_idx];
    //     *(uint16_t*)(&(is_ch1 ? chunk1 : chunk2)[i * chunk_stride_c * 2 + threadid]) = val;
    // }
    // // __syncthreads();
    
    // Declare the fragments
    // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> had_frag[2];
    b32 had_frag[8];
    int remaining_log_had_size2 = log_had_size;
    // __shared__ b32 shuffle[32 * warps_per_block]; // TODO: consider * 2 because loading 2 hadamards
    // __shared__ b32 shuffle2[num_chunks][32 * warps_per_block];
    // __shared__ half temp_had[16 * 16 * 2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int c_log_h = min(4, remaining_log_had_size2);
        b32* curr_had = (b32*)&hads[256 * (c_log_h - 1)];

        int col = (threadid % 4) + (threadid / 16) * 4;//*2 from 32-bit/16-bit
        int row = (threadid % 16) / 4;
        #pragma unroll
        for (int x = 0; x < 2; x++) {
            #pragma unroll
            for (int y = 0; y < 2; y++) {
                b32* a_loc = &((b32*)curr_had)[col + (row + y * 4 + x * 8) * 8];
                asm(
                    "ld.global.L2::256B.u32 %0, [%1];\n" : "=r"(had_frag[i * 4 + x * 2 + y]) : "l"(a_loc)
                );
            }
            // need ternary operator so that indices are known at compile time
            // shuffle[threadIdx.x] = threadid < 16 ? had_frag[i * 4 + x * 2 + 1] : had_frag[i * 4 + x * 2 + 0];
            // if (threadid < 16)
            //     had_frag[i * 4 + x * 2 + 1] = shuffle[threadIdx.x ^ 16];
            // else
            //     had_frag[i * 4 + x * 2 + 0] = shuffle[threadIdx.x ^ 16];
            b32 swap = threadid < 16 ? had_frag[i * 4 + x * 2 + 1] : had_frag[i * 4 + x * 2 + 0]; // bool acts as 0 or 1 (we are true C++ programmers)
            swap = __shfl_xor_sync(0xFFFFFFFF, swap, 16);
            if (threadid < 16)
                had_frag[i * 4 + x * 2 + 1] = swap;
            else
                had_frag[i * 4 + x * 2 + 0] = swap;
        }
        // NOTE: because it's a Walsh matrix, 
        // [a00, a01] = [a00, a10]
        // [a10, a11]   [a01, a11]
        // so the following swap isn't necessary
        // b32 swap2 = had_frag[i * 4 + 1];
        // had_frag[i * 4 + 1] = had_frag[i * 4 + 2];
        // had_frag[i * 4 + 2] = swap2;
        
        // int row = threadid / 4;
        // int col = threadid % 4;//*2 from 32-bit/16-bit
        // #pragma unroll
        // for (int j = 0; j < 4; j++) {
        //     had_frag[i * 4 + j] = curr_had[(row + (j % 2) * 8) * 8 + (col + (j / 2) * 4)];
        // }


        // wmma::load_matrix_sync(had_frag[i], &hads[256 * (c_log_h - 1)], 16);
        
        remaining_log_had_size2 -= 4;
        if(remaining_log_had_size2 <= 0) break;
    }

    #pragma unroll
    for (int k = 0; k < num_chunks; k++) {
        // if (k < num_chunks - 1) {
            // #pragma unroll
            // for(int j = 0; j < 16; j += 2){
            //     uint32_t a_idx = j * stride_c + threadid * stride_r + 256 * (k + 1);
            //     half* a_loc = a + a_idx;
            //     uint16_t val;
            //     asm(
            //         "ld.global.L1::no_allocate.L2::256B.u16 %0, [%1];\n" : "=h"(val) : "l"(a_loc)
            //     );
            //     *(uint16_t*)(&(!is_ch1 ? chunk1 : chunk2)[j * chunk_stride_c + threadid]) = val;
            //     // (!is_ch1 ? chunk1 : chunk2)[j * chunk_stride_c + threadid] = a_ann[j * stride_c + threadid * stride_r + 256 * (k + 1)]; // + 256 * (k + 1) assumes a is column-major
            // }
        // }

        register b32 b_frag[4];
        // load b_frag
        int row = (threadid % 4) + (threadid / 16) * 4;//*2 from 32-bit/16-bit
        int col = (threadid % 16) / 4;
        #pragma unroll
        for (int x = 0; x < 2; x++) {
            #pragma unroll
            for (int y = 0; y < 2; y++) {
                b32* a_loc = &((b32*)a)[row + (col + y * 4 + x * 8) * 8];
                asm(
                    "ld.global.L2::256B.u32 %0, [%1];\n" : "=r"(b_frag[x * 2 + y]) : "l"(a_loc)
                );
            }
            // shuffle2[k][threadIdx.x] = threadid < 16 ? b_frag[x * 2 + 1] : b_frag[x * 2 + 0];
            b32 swap = threadid < 16 ? b_frag[x * 2 + 1] : b_frag[x * 2 + 0]; // bool acts as 0 or 1 (we are true C++ programmers)
            swap = __shfl_xor_sync(0xFFFFFFFF, swap, 16);
            if (threadid < 16)
                b_frag[x * 2 + 1] = swap; //shuffle2[k][threadIdx.x ^ 16];
            else
                b_frag[x * 2 + 0] = swap; //shuffle2[k][threadIdx.x ^ 16];
        }

        // for (int j = 0; j < 4; j++) {
        //     b32* a_loc = &((b32*)a)[(row + (j % 2) * 4) + (col + (j / 2) * 8) * 8];
        //     // b_frag[j] = ((b32*)a)[(row + (j % 2) * 4) + (col + (j / 2) * 8) * 8]; // TODO: use the ld no alloc thing
        //     asm(
        //         "ld.global.L2::256B.u32 %0, [%1];\n" : "=r"(b_frag[j]) : "l"(a_loc)
        //     ); // .L1::no_allocate
        // }

        int remaining_log_had_size = log_had_size;

        #pragma unroll
        for(int i = 0; i < 2; i++) {
            int c_log_h = min(4, remaining_log_had_size);

            // // Define the fragments
            // wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            // wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
            // // Initialize the output to zero
            // wmma::fill_fragment(c_frag, 0.0f);
            // // Load the inputs
            // wmma::load_matrix_sync(b_frag, (is_ch1 ? chunk1 : chunk2), chunk_stride_c);
            // // Perform the matrix multiplication
            // wmma::mma_sync(c_frag, had_frag[i], b_frag, c_frag);


            mma_m16_n16_k16_fp16_fp16_fp16_noacc(had_frag[i * 4 + 0], had_frag[i * 4 + 1], had_frag[i * 4 + 2], had_frag[i * 4 + 3], b_frag[0], b_frag[1], b_frag[2], b_frag[3], b_frag[0], b_frag[1], b_frag[2], b_frag[3]);


            remaining_log_had_size -= 4;
            // wmma::layout_t mode = (remaining_log_had_size <= 0 && i == 0) ? wmma::mem_col_major : wmma::mem_row_major;
            // wmma::store_matrix_sync((is_ch1 ? chunk1 : chunk2), c_frag, chunk_stride_c, mode); //col_major); // we store row major to get a free transpose

            if (remaining_log_had_size <= 0 && i == 0) {
                // TODO: consider different storing so no need for transpose
                matrix_transpose_m8_n8_fp16_inplace(b_frag[0]);
                matrix_transpose_m8_n8_fp16_inplace(b_frag[1]);
                matrix_transpose_m8_n8_fp16_inplace(b_frag[2]);
                matrix_transpose_m8_n8_fp16_inplace(b_frag[3]);
            } else {
                // swap and use output directly as b_frag for next iteration as an actually free transpose
                b32 temp = b_frag[1];
                b_frag[1] = b_frag[2];
                b_frag[2] = temp;
            }

            if(remaining_log_had_size <= 0) break;
        }

        // cuda::access_property ap(cuda::access_property::streaming{});
        // cuda::annotated_ptr<half, cuda::access_property> a_ann{a, ap}; 
        // #pragma unroll
        // for(int j = 0; j < 16; j += 2){
        //     a[j * stride_c + threadid * stride_r + 256 * k] = (is_ch1 ? chunk1 : chunk2)[j * chunk_stride_c + threadid];
        // }

        #pragma unroll
        for (int x = 0; x < 2; x++) {
            b32 swap = b_frag[x * 2 + (threadid < 16)]; // bool acts as 0 or 1 (we are true C++ programmers)
            swap = __shfl_xor_sync(0xFFFFFFFF, swap, 16);
            b_frag[x * 2 + (threadid < 16)] = swap;

            #pragma unroll
            for (int y = 0; y < 2; y++) {
                b32* a_loc = &((b32*)a)[row + (col + y * 4 + x * 8) * 8];
                asm(
                    "st.global.L1::evict_first.u32 [%0], %1;\n" :: "l"(a_loc), "r"(b_frag[x * 2 + y]) // : "memory" // NOTE: don't need since writing full cache line and don't access this mem afterward
                );
            }
        }

        // int newrow = threadid % 4;
        // int newcol = threadid / 4;

        // #pragma unroll
        // for (int j = 0; j < 4; j++) {
        //     ((b32*)a)[(newrow + (j % 2) * 4) + (newcol + (j / 2) * 8) * 8] = b_frag[j];
        // }
        a += 256; // move on to next chunk by skipping 256 elements in fp16

        // half* temp = chunk;
        // chunk = next_chunk;
        // next_chunk = temp;
        // is_ch1 = !is_ch1;
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
    uint32_t cols = 4096 * 16;
    constexpr int chunks_per_warp = 2;
    constexpr int warps_per_block = 2;

    half* ptr = (half*)malloc(vector_size * cols * sizeof(half)); // col major

    for(int i = 0; i < vector_size * cols; i++) ptr[i] = 1;//(i % cols == i / cols);//1;
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

    half rsqrt2 = (half)(1/sqrt(2));

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