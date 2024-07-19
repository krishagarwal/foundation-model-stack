#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#include <cuda_fp16.h>
#include <cuda/annotated_ptr>
#include <pybind11/pybind11.h>

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

#define p_p(i) ((fp16_1p[i] & 0x0000FFFF) | fp16_1p[i] << 16)
#define p_n(i) ((fp16_1p[i] & 0x0000FFFF) | fp16_1n[i] << 16)
#define n_p(i) ((fp16_1n[i] & 0x0000FFFF) | fp16_1p[i] << 16)
#define n_n(i) ((fp16_1n[i] & 0x0000FFFF) | fp16_1n[i] << 16)

template<int num_chunks, int warps_per_block, int log_had_size>
__global__ void __launch_bounds__(32 * warps_per_block, MIN(24, 48 / warps_per_block))
// a is column major, b is row major
wmma_ker(half* __restrict__ a, half* __restrict__ hads, uint32_t stride_r, uint32_t stride_c) {//, int log_had_size) {
    uint blockid = blockIdx.x * warps_per_block + threadIdx.x / 32;
    uint threadid = threadIdx.x % 32;
    a += blockid * stride_c * 16 * num_chunks; // swapped blockidx.x,y cause 64k limit
    
    constexpr uint16_t fp16_1p[4] = {0b0011100110101000, 0b0011100000000000, 0b0011010110101000, 0b0011010000000000};// 0b0011110000000000;
    constexpr uint16_t fp16_1n[4] = {0b1011100110101000, 0b1011100000000000, 0b1011010110101000, 0b1011010000000000};// 0b1011110000000000;
    constexpr b32 p_p[4] = {p_p(0), p_p(1), p_p(2), p_p(3)};
    constexpr b32 p_n[4] = {p_n(0), p_n(1), p_n(2), p_n(3)};
    constexpr b32 n_p[4] = {n_p(0), n_p(1), n_p(2), n_p(3)};
    constexpr b32 n_n[4] = {n_n(0), n_n(1), n_n(2), n_n(3)};
    // const b32 had16[4 * 32] = {
    //     p_p,p_p,p_p,p_p,p_p,p_p,p_p,p_p,
    //     p_n,p_n,p_n,p_n,p_n,p_n,p_n,p_n,
    //     p_p,n_n,p_p,n_n,p_p,n_n,p_p,n_n,
    //     p_n,n_p,p_n,n_p,p_n,n_p,p_n,n_p,
    //     p_p,p_p,n_n,n_n,p_p,p_p,n_n,n_n,
    //     p_n,p_n,n_p,n_p,p_n,p_n,n_p,n_p,
    //     p_p,n_n,n_n,p_p,p_p,n_n,n_n,p_p,
    //     p_n,n_p,n_p,p_n,p_n,n_p,n_p,p_n,
    //     p_p,p_p,p_p,p_p,n_n,n_n,n_n,n_n,
    //     p_n,p_n,p_n,p_n,n_p,n_p,n_p,n_p,
    //     p_p,n_n,p_p,n_n,n_n,p_p,n_n,p_p,
    //     p_n,n_p,p_n,n_p,n_p,p_n,n_p,p_n,
    //     p_p,p_p,n_n,n_n,n_n,n_n,p_p,p_p,
    //     p_n,p_n,n_p,n_p,n_p,n_p,p_n,p_n,
    //     p_p,n_n,n_n,p_p,n_n,p_p,p_p,n_n,
    //     p_n,n_p,n_p,p_n,n_p,p_n,p_n,n_p,
    // };
    // const uint32_t had16_bools[8] = {
    //     0b11111111111111111010101010101010,
    //     0b11001100110011001001100110011001,
    //     0b11110000111100001010010110100101,
    //     0b11000011110000111001011010010110,
    //     0b11111111000000001010101001010101,
    //     0b11001100001100111001100101100110,
    //     0b11110000000011111010010101011010,
    //     0b11000011001111001001011001101001,
    // };
    const b32 had_16_p1[4][4] = {
        {
            0b10001000010001000010001000010001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b10001000010001000010001000010001
        },
        {
            0b11001100100010000011001100100010,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11001100100010000011001100100010
        },
        {
            0b11111111101010101100110010011001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11111111101010101100110010011001
        },
        {
            0b11111111101010101100110010011001,
            0b11111111101010101100110010011001,
            0b11111111101010101100110010011001,
            0b00000000010101010011001101100110
        }
    };
    const b32 had_16_p2[4][4] = {
        {
            0b10000000010000000010000000010000,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b10000000010000000010000000010000
        },
        {
            0b11000000100001000011000000100001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11000000100001000011000000100001
        },
        {
            0b11110000101001011100001110010110,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11110000101001011100001110010110
        },
        {
            0b11110000101001011100001110010110,
            0b11110000101001011100001110010110,
            0b11110000101001011100001110010110,
            0b00001111010110100011110001101001
        }
    };

    const b32 had_16_mask[3][4] = {
        {
            0b10001000010001000010001000010001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b10001000010001000010001000010001
        },
        {
            0b11001100110011000011001100110011,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11001100110011000011001100110011
        },
        {
            0b11111111111111111111111111111111,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11111111111111111111111111111111
        }
    };

    b32 had_frag[8];
    int remaining_log_had_size2 = log_had_size;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int c_log_h = min(4, remaining_log_had_size2);

        // b32* curr_had = (b32*)&hads[256 * (c_log_h - 1)];
        // int row = threadid / 4;
        // int col = threadid % 4;//*2 from 32-bit/16-bit
        // #pragma unroll
        // for (int j = 0; j < 4; j++) {
        //     had_frag[i * 4 + j] = had16[(row + (j % 2) * 8) * 8 + (col + (j / 2) * 4)]; //curr_had[(row + (j % 2) * 8) * 8 + (col + (j / 2) * 4)];
        // }
        // #pragma unroll
        // for (int t = 0; t < 32; t++){
        //     if (threadid == t) {
        //         int row = t / 4;
        //         int col = t % 4;
        //         #pragma unroll
        //         for (int j = 0; j < 4; j++) {
        //             had_frag[i * 4 + j] = had16[(row + (j % 2) * 8) * 8 + (col + (j / 2) * 4)];
        //         }
        //     }
        // }
        // int row = threadid / 4;
        // int col = threadid % 4;
        // #pragma unroll
        // for (int j = 0; j < 4; j++) {
        //     int index = (row + (j % 2) * 8) * 8 + (col + (j / 2) * 4);
        //     int index1 = index * 2 + 0;
        //     int index2 = index * 2 + 1;
        //     bool pred1 = had16_bools[index1 / 32] & (1 << (31 - (index1 % 32)));
        //     bool pred2 = had16_bools[index2 / 32] & (1 << (31 - (index2 % 32)));
        //     b32 val = pred1 ? (pred2 ? p_p : p_n) : (pred2 ? n_p : n_n);
        //     had_frag[i * 4 + j] = val;// had16[(row + (j % 2) * 8) * 8 + (col + (j / 2) * 4)];
        // }
        // int row = threadid / 4;
        // int col = threadid % 4;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (c_log_h < 4) {
                bool mask = had_16_mask[c_log_h - 1][j] & (1 << (31 - threadid));
                if (!mask) {
                    had_frag[i * 4 + j] = 0;
                    continue;
                }
            }
            bool pred1 = had_16_p1[c_log_h - 1][j] & (1 << (31 - threadid));
            bool pred2 = had_16_p2[c_log_h - 1][j] & (1 << (31 - threadid));
            b32 val = pred1 ? (pred2 ? p_p[c_log_h - 1] : p_n[c_log_h - 1]) : (pred2 ? n_p[c_log_h - 1] : n_n[c_log_h - 1]);
            had_frag[i * 4 + j] = val;// had16[(row + (j % 2) * 8) * 8 + (col + (j / 2) * 4)];
        }
        // int row = threadid / 4;
        // int col = threadid % 4;
        // #pragma unroll
        // for (int j = 0; j < 4; j++) {
        //     bool pred1 = had_16_p1[j] & (1 << threadid);
        //     bool pred2 = had_16_p2[j] & (1 << threadid);
        //     b32 val = pred1 ? (pred2 ? p_p : p_n) : (pred2 ? n_p : n_n);
        //     had_frag[i * 4 + j] = 1;//val;//had16[(row + (j % 2) * 8) * 8 + (col + (j / 2) * 4)];
        // }

        remaining_log_had_size2 -= 4;
        if(remaining_log_had_size2 <= 0) break;
    }


    #pragma unroll
    for (int k = 0; k < num_chunks; k++) {
        register b32 b_frag[4];
        int row = threadid % 4;//*2 from 32-bit/16-bit
        int col = threadid / 4;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            b32* a_loc = &((b32*)a)[(row + (j % 2) * 4) + (col + (j / 2) * 8) * 8];
            asm(
                "ld.global.L2::256B.u32 %0, [%1];\n" : "=r"(b_frag[j]) : "l"(a_loc)
            );
        }

        int remaining_log_had_size = log_had_size;

        #pragma unroll
        for(int i = 0; i < 2; i++) {
            int c_log_h = min(4, remaining_log_had_size);
            mma_m16_n16_k16_fp16_fp16_fp16_noacc(had_frag[i * 4 + 0], had_frag[i * 4 + 1], had_frag[i * 4 + 2], had_frag[i * 4 + 3], b_frag[0], b_frag[1], b_frag[2], b_frag[3], b_frag[0], b_frag[1], b_frag[2], b_frag[3]);

            remaining_log_had_size -= 4;
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

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            ((b32*)a)[(row + (j % 2) * 4) + (col + (j / 2) * 8) * 8] = b_frag[j];
        }

        a += 256; // move on to next chunk by skipping 256 elements in fp16
    }
}

void fast_had_trans(uint64_t a, uint32_t M, uint32_t N, uint32_t log_had_size) {
    void* a_mat = (void*) a;
    uint32_t num_chunks = (M * N + 255) / 256;
    if ((M * N) % 256 != 0) {
        size_t old_size = M * N * sizeof(half), new_size = num_chunks * 256 * sizeof(half);
        cudaMalloc(&a_mat, new_size);
        cudaMemcpy(a_mat, (void*) a, old_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        cudaMemset((half*) a_mat + M * N, 0, new_size - old_size);
    }

    constexpr int chunks_per_warp = 2;
    constexpr int warps_per_block = 1;
    constexpr int block_size = 32 * warps_per_block;
    int grid_size = num_chunks / chunks_per_warp / warps_per_block;
    switch (log_had_size) {
        case 1: wmma_ker<chunks_per_warp, warps_per_block, 1><<<dim3(grid_size), dim3(block_size)>>>((half*) a_mat, NULL, 1, 16); break;
        case 2: wmma_ker<chunks_per_warp, warps_per_block, 2><<<dim3(grid_size), dim3(block_size)>>>((half*) a_mat, NULL, 1, 16); break;
        case 3: wmma_ker<chunks_per_warp, warps_per_block, 3><<<dim3(grid_size), dim3(block_size)>>>((half*) a_mat, NULL, 1, 16); break;
        case 4: wmma_ker<chunks_per_warp, warps_per_block, 4><<<dim3(grid_size), dim3(block_size)>>>((half*) a_mat, NULL, 1, 16); break;
        case 5: wmma_ker<chunks_per_warp, warps_per_block, 5><<<dim3(grid_size), dim3(block_size)>>>((half*) a_mat, NULL, 1, 16); break;
        case 6: wmma_ker<chunks_per_warp, warps_per_block, 6><<<dim3(grid_size), dim3(block_size)>>>((half*) a_mat, NULL, 1, 16); break;
        case 7: wmma_ker<chunks_per_warp, warps_per_block, 7><<<dim3(grid_size), dim3(block_size)>>>((half*) a_mat, NULL, 1, 16); break;
        case 8: wmma_ker<chunks_per_warp, warps_per_block, 8><<<dim3(grid_size), dim3(block_size)>>>((half*) a_mat, NULL, 1, 16); break;
    }

    if ((uint64_t) a_mat != a) {
        cudaMemcpy((void*) a, a_mat, M * N * sizeof(half), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        cudaFree(a_mat);
    }
}

// template <typename T>
int main()
{
    // py::print("hi", true);
    printf("hi\n");

    // 16x4096
    uint32_t vector_size = 256;
    constexpr uint32_t log_had_size = 6;
    uint32_t cols = 4096 * 16;
    constexpr int chunks_per_warp = 2;
    constexpr int warps_per_block = 1;

    half* ptr = (half*)malloc(vector_size * cols * sizeof(half)); // col major

    for(int i = 0; i < vector_size * cols; i++) ptr[i] = 1;//(i % (1 << log_had_size) == i / (1 << log_had_size));
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

namespace py = pybind11;
PYBIND11_MODULE(tensor_core_had, m) {
    m.doc() = "fast hadamard transform module";
    m.def("fast_had_trans", &fast_had_trans, "A function to perform a fast hadamard transform", py::arg("a"), py::arg("M"), py::arg("N"), py::arg("log_had_size"));
}