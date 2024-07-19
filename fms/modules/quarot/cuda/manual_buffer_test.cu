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

typedef uint32_t b32;

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

template<int num_chunks, int warps_per_block, int log_had_size>
__global__ void __launch_bounds__(32 * warps_per_block, MIN(24, 48 / warps_per_block))
// a is column major, b is row major
wmma_ker(half* __restrict__ a, half* __restrict__ hads, uint32_t stride_r, uint32_t stride_c) {//, int log_had_size) {
    uint blockid = blockIdx.x * warps_per_block + threadIdx.x / 32;
    uint threadid = threadIdx.x % 32;
    a += blockid * stride_c * 16 * num_chunks; // swapped blockidx.x,y cause 64k limit

    b32 b_frag[4 * num_chunks];
    int row = threadid % 4;//*2 from 32-bit/16-bit
    int col = threadid / 4;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        b32* a_loc = &((b32*)a)[(row + (j % 2) * 4) + (col + (j / 2) * 8) * 8];
        asm(
            "ld.global.L2::256B.u32 %0, [%1];\n" : "=r"(b_frag[j]) : "l"(a_loc)
        );
    }
    
    constexpr uint16_t fp16_1p = 0b0011010000000000;// 0b0011110000000000;
    constexpr uint16_t fp16_1n = 0b1011010000000000;// 0b1011110000000000;
    constexpr b32 p_p = (fp16_1p & 0x0000FFFF) | fp16_1p << 16;
    constexpr b32 p_n = (fp16_1p & 0x0000FFFF) | fp16_1n << 16;
    constexpr b32 n_p = (fp16_1n & 0x0000FFFF) | fp16_1p << 16;
    constexpr b32 n_n = (fp16_1n & 0x0000FFFF) | fp16_1n << 16;

    const b32 had_16_p1[4] = {
        0b11111111101010101100110010011001,
        0b11111111101010101100110010011001,
        0b11111111101010101100110010011001,
        0b00000000010101010011001101100110,
    };
    const b32 had_16_p2[4] = {
        0b11110000101001011100001110010110,
        0b11110000101001011100001110010110,
        0b11110000101001011100001110010110,
        0b00001111010110100011110001101001,
    };

    b32 had_frag[8];
    int remaining_log_had_size2 = log_had_size;
    #pragma unroll
    for (int i = 0; i < 2; i++) {

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            bool pred1 = had_16_p1[j] & (1 << (31 - threadid));
            bool pred2 = had_16_p2[j] & (1 << (31 - threadid));
            b32 val = pred1 ? (pred2 ? p_p : p_n) : (pred2 ? n_p : n_n);
            had_frag[i * 4 + j] = val;// had16[(row + (j % 2) * 8) * 8 + (col + (j / 2) * 4)];
        }

        remaining_log_had_size2 -= 4;
        if(remaining_log_had_size2 <= 0) break;
    }

    #pragma unroll
    for (int k = 0; k < num_chunks; k++) {
        if (k + 1 < num_chunks) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                b32* a_loc = &((b32*)(a + 256))[(row + (j % 2) * 4) + (col + (j / 2) * 8) * 8];
                asm(
                    "ld.global.L2::256B.u32 %0, [%1];\n" : "=r"(b_frag[j + 4 * (k + 1)]) : "l"(a_loc)
                );
            }
        }

        int remaining_log_had_size = log_had_size;

        #pragma unroll
        for(int i = 0; i < 2; i++) {
            int c_log_h = min(4, remaining_log_had_size);
            mma_m16_n16_k16_fp16_fp16_fp16_noacc(had_frag[i * 4 + 0], had_frag[i * 4 + 1], had_frag[i * 4 + 2], had_frag[i * 4 + 3], b_frag[0 + 4 * k], b_frag[1 + 4 * k], b_frag[2 + 4 * k], b_frag[3 + 4 * k], b_frag[0 + 4 * k], b_frag[1 + 4 * k], b_frag[2 + 4 * k], b_frag[3 + 4 * k]);

            remaining_log_had_size -= 4;
            if (remaining_log_had_size <= 0 && i == 0) {
                // TODO: consider different storing so no need for transpose
                matrix_transpose_m8_n8_fp16_inplace(b_frag[0 + 4 * k]);
                matrix_transpose_m8_n8_fp16_inplace(b_frag[1 + 4 * k]);
                matrix_transpose_m8_n8_fp16_inplace(b_frag[2 + 4 * k]);
                matrix_transpose_m8_n8_fp16_inplace(b_frag[3 + 4 * k]);
            } else {
                // swap and use output directly as b_frag for next iteration as an actually free transpose
                b32 temp = b_frag[1 + 4 * k];
                b_frag[1 + 4 * k] = b_frag[2 + 4 * k];
                b_frag[2 + 4 * k] = temp;
            }

            if(remaining_log_had_size <= 0) break;
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            ((b32*)a)[(row + (j % 2) * 4) + (col + (j / 2) * 8) * 8] = b_frag[j + 4 * k];
        }

        a += 256; // move on to next chunk by skipping 256 elements in fp16
    }
}


// template <typename T>
int main()
{
    // py::print("hi", true);
    printf("hi\n");

    // 16x4096
    uint32_t vector_size = 256;
    constexpr uint32_t log_had_size = 8;
    uint32_t cols = 4096 * 16;
    constexpr int chunks_per_warp = 2;
    constexpr int warps_per_block = 1;

    half* ptr = (half*)malloc(vector_size * cols * sizeof(half)); // col major

    for(int i = 0; i < vector_size * cols; i++) ptr[i] = (i % (1 << log_had_size) == i / (1 << log_had_size));
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
    wmma_ker<chunks_per_warp, warps_per_block, log_had_size><<<dim3(((cols / 16) * (vector_size / 16)) / chunks_per_warp / warps_per_block, 1), dim3(32 * warps_per_block)>>>(dev_ptr, dev_had_16, 1, 16);//, log_had_size);
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