#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cuda_fp16.h>
#include <cuda/annotated_ptr>
#ifndef NOPYBIND // use -DNOPYBIND to compile without pybind11
#include <pybind11/pybind11.h>
#endif


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

template <int n>
static void inline wait_group() {
    asm("cp.async.wait_group %0;\n" :: "n"(n));
}

#define p_p(i) ((fp16_1p[i] & 0x0000FFFF) | fp16_1p[i] << 16)
#define p_n(i) ((fp16_1p[i] & 0x0000FFFF) | fp16_1n[i] << 16)
#define n_p(i) ((fp16_1n[i] & 0x0000FFFF) | fp16_1p[i] << 16)
#define n_n(i) ((fp16_1n[i] & 0x0000FFFF) | fp16_1n[i] << 16)

template<int num_chunks, int warps_per_block, int log_had_size>
// TODO: check if blocks per SM is correctly calculated
__global__ void __launch_bounds__(32 * warps_per_block, MIN(MIN((cudaSharedmemCarveoutMaxShared * 1024) / (num_chunks * warps_per_block * 128 * 4), 24), 48 / warps_per_block))//MIN(24, 48 / warps_per_block))
// a is column major, b is row major
wmma_ker(half* __restrict__ a) {
    uint blockid = blockIdx.x * warps_per_block + threadIdx.x / 32;
    uint threadid = threadIdx.x % 32;
    extern __shared__ b32 bfrag_arr[]; // num_chunks * warps_per_block * 128

    half* orig_a = a;
    a += blockid * num_chunks * 256;
    b32* a_ptr = ((b32*) a) + threadid * 4;
    b32* b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * 128 + threadid * 4;
    #pragma unroll
    for (int k = 0; k < num_chunks; k++) {
        size_t shared_ptr = __cvta_generic_to_shared(b_frag_ptr);
        asm volatile(
            "cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\n"
            "cp.async.commit_group;\n"
            :: "l"(shared_ptr), "l"(a_ptr)
        );
        a_ptr += 128;
        b_frag_ptr += 128;
    }

    // generate hadamard 16x16 (up to 2 of them)
    constexpr uint16_t fp16_1p[4] = {0b0011100110101000, 0b0011100000000000, 0b0011010110101000, 0b0011010000000000};// 0b0011110000000000;
    constexpr uint16_t fp16_1n[4] = {0b1011100110101000, 0b1011100000000000, 0b1011010110101000, 0b1011010000000000};// 0b1011110000000000;
    constexpr b32 p_p[4] = {p_p(0), p_p(1), p_p(2), p_p(3)};
    constexpr b32 p_n[4] = {p_n(0), p_n(1), p_n(2), p_n(3)};
    constexpr b32 n_p[4] = {n_p(0), n_p(1), n_p(2), n_p(3)};
    constexpr b32 n_n[4] = {n_n(0), n_n(1), n_n(2), n_n(3)};
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
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int c_log_h = (i == 0) ? min(4, log_had_size) : log_had_size % 4;
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
            had_frag[i * 4 + j] = val;
        }
        if (log_had_size <= 4 || log_had_size % 4 == 0) break;
    }
    
    // do the multiplication
    int row = threadid % 4;
    int col = threadid / 4;
    
    #pragma unroll
    for (int l = 0; l < 2; l++) { // TODO: debug
        // max is just to avoid warning. This is only used if l == 1, but that only happens if log_had_size > 8
        int iter_2_log_had_size = max(0, log_had_size - 8); 
        // TODO: stride?
        b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * (l == 0 ? 128 : (128 >> (iter_2_log_had_size))); // TODO: works for >=16(?) but not for <16 remaining had size
        #pragma unroll
        for (int k = 0; k < num_chunks; k++) {
            b32 b_frag[4];
            
            if (l == 0) {
                // TODO: bad fix for k not being recognized as a constexpr
                switch(k) {
                    case 0: asm volatile("cp.async.wait_group %0;\n" :: "n"(num_chunks - 1)); break;
                    case 1: asm volatile("cp.async.wait_group %0;\n" :: "n"(num_chunks - 2)); break;
                    case 2: asm volatile("cp.async.wait_group %0;\n" :: "n"(num_chunks - 3)); break;
                    case 3: asm volatile("cp.async.wait_group %0;\n" :: "n"(num_chunks - 4)); break;
                    case 4: asm volatile("cp.async.wait_group %0;\n" :: "n"(num_chunks - 5)); break;
                    case 5: asm volatile("cp.async.wait_group %0;\n" :: "n"(num_chunks - 6)); break;
                    case 6: asm volatile("cp.async.wait_group %0;\n" :: "n"(num_chunks - 7)); break;
                    case 7: asm volatile("cp.async.wait_group %0;\n" :: "n"(num_chunks - 8)); break;
                }
                // asm("cp.async.wait_group %0;\n" :: "n"(num_chunks - k - 1));
            }

            int pending_log_had_size = log_had_size - l * 8;

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (l == 0) {
                    b_frag[j] = b_frag_ptr[(row + (j % 2) * 4) + (col + (j / 2) * 8) * 8];
                } else {
                    if (pending_log_had_size < 4) {
                        int small_factor = 1 << (4 - pending_log_had_size); // 16 / remaning_size
                        int col_small = col * small_factor; // need to increment col more
                        int colidx = col_small;
                        int rowidx = 2 * row + (j % 2) + 8 * (j / 2);
                        int rowidx_good = rowidx % (1 << pending_log_had_size); // portion of rowidx in bounds
                        int rowidx_mult = rowidx / (1 << pending_log_had_size); // number of times to increment col
                        // TODO: stride
                        b_frag[j] = b_frag_ptr[rowidx_good * 128 + colidx + rowidx_mult];
                    } else {
                        int colidx = col >> (pending_log_had_size - 4);
                        int rowidx = 2 * row + 16 * (col % (1 << (pending_log_had_size - 4))) + (j % 2) + 8 * (j / 2);
                        // TODO: stride
                        b_frag[j] = b_frag_ptr[rowidx * 128 + colidx];
                    }
                }
            }

            if (l == 1) {
                b32 f0 = ((b_frag[1] & 0xFFFF) << 16) | (b_frag[0] & 0xFFFF);
                b32 f1 = ((b_frag[3] & 0xFFFF) << 16) | (b_frag[2] & 0xFFFF);
                b32 f2 = (b_frag[1] & 0xFFFF0000) | (b_frag[0] >> 16);
                b32 f3 = (b_frag[3] & 0xFFFF0000) | (b_frag[2] >> 16);
                b_frag[0] = f0;
                b_frag[1] = f1;
                b_frag[2] = f2;
                b_frag[3] = f3;
            }

            int remaining_log_had_size = log_had_size - l * 8;

            #pragma unroll
            for(int i = 0; i < 2; i++) {
                int had_off = ((remaining_log_had_size < 4) && !(log_had_size <= 4 || log_had_size % 4 == 0)) ? 4 : 0;
                mma_m16_n16_k16_fp16_fp16_fp16_noacc(had_frag[had_off + 0], had_frag[had_off + 1], had_frag[had_off + 2], had_frag[had_off + 3], b_frag[0], b_frag[1], b_frag[2], b_frag[3], b_frag[0], b_frag[1], b_frag[2], b_frag[3]);

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

            if (l == 1) {
                b32 f0 = ((b_frag[2] & 0xFFFF) << 16) | (b_frag[0] & 0xFFFF);
                b32 f1 = (b_frag[2] & 0xFFFF0000) | (b_frag[0] >> 16);
                b32 f2 = ((b_frag[3] & 0xFFFF) << 16) | (b_frag[1] & 0xFFFF);
                b32 f3 = (b_frag[3] & 0xFFFF0000) | (b_frag[1] >> 16);
                b_frag[0] = f0;
                b_frag[1] = f1;
                b_frag[2] = f2;
                b_frag[3] = f3;
            }

            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (l == 0) {
                    b32* store = (log_had_size <= 8) ? (b32*)a : b_frag_ptr;
                    // b32* store = (b32*)a; // TODO: debug
                    store[(row + (j % 2) * 4) + (col + (j / 2) * 8) * 8] = b_frag[j];
                } else {
                    // TODO: stride
                    b32* store = (b32*)(orig_a + (blockid / warps_per_block) * (num_chunks * warps_per_block) * 256 + (256 >> (iter_2_log_had_size)) * (num_chunks * (blockid % warps_per_block) + k));
                    if (pending_log_had_size < 4) {
                        int small_factor = 1 << (4 - pending_log_had_size); // 16 / remaning_size
                        int col_small = col * small_factor; // need to increment col more
                        int colidx = col_small;
                        int rowidx = 2 * row + (j % 2) + 8 * (j / 2);
                        int rowidx_good = rowidx % (1 << pending_log_had_size); // portion of rowidx in bounds
                        int rowidx_mult = rowidx / (1 << pending_log_had_size); // number of times to increment col
                        // TODO: stride
                        // store[rowidx_good * 128 + colidx + rowidx_mult] = b_frag[j];
                        b_frag_ptr[rowidx_good * 128 + colidx + rowidx_mult] = b_frag[j];
                    } else {
                        // int colidx = col >> (pending_log_had_size - 4);
                        // int rowidx = 2 * row + 16 * (col % (1 << (pending_log_had_size - 4))) + (j % 2) + 8 * (j / 2);
                        // // TODO: stride
                        // store[rowidx * 128 + colidx] = b_frag[j];

                        int colidx = col >> (pending_log_had_size - 4);
                        int rowidx = 2 * row + 16 * (col % (1 << (pending_log_had_size - 4))) + (j % 2) + 8 * (j / 2);
                        // TODO: stride
                        b_frag_ptr[rowidx * 128 + colidx] = b_frag[j];

                        // int col2 = threadid % 8;
                        // int row2 = threadid / 8;
                        // int colidx = col2 >> (pending_log_had_size - 4);
                        // int rowidx = 2 * row2 + 16 * (col2 % (1 << (pending_log_had_size - 4))) + (j % 2) + 8 * (j / 2);
                        // // shuffle
                        // int target = (threadid % 8) * 4 + (threadid / 8);
                        // b_frag[j] = __shfl_sync(0xFFFFFFFF, b_frag[j], target);
                        // store[rowidx * 128 + colidx] = b_frag[j];
                    }
                }
            }

            a += 256; // (only affects first 256 size) move on to next chunk by skipping 256 elements in fp16
            // TODO: stride
            b_frag_ptr += (l == 0 ? 128 : (128 >> (iter_2_log_had_size)));
        }
        if (log_had_size <= 8)
            break;
        if (l == 0)
            __syncthreads();
    }

    // if did second iteration, store here
    if (log_had_size > 8) {
        __syncthreads();
        half* a_ptr_half = orig_a + (num_chunks * 256) * blockid;
        b32* a_ptr = (b32*) a_ptr_half + threadid;// + threadid * 4;
        b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * 128 + threadid;// + threadid * 4;
        #pragma unroll
        for (int k = 0; k < num_chunks; k++) {
            a_ptr[0] = b_frag_ptr[0];
            a_ptr[32] = b_frag_ptr[32];
            a_ptr[64] = b_frag_ptr[64];
            a_ptr[96] = b_frag_ptr[96];
            a_ptr += 128;
            b_frag_ptr += 128;
        }
    }

}


template <int chunks_per_warp, int warps_per_block, int log_had_size>
void __forceinline__ run_matmul_kernel (half* a_mat, int num_chunks) {
    int shared_size = chunks_per_warp * warps_per_block * 128 * 4;
    dim3 grid_size = num_chunks / chunks_per_warp / warps_per_block;
    dim3 block_size = 32 * warps_per_block;
    if (shared_size > 48 * 1024) {
        void* func_ptr = (void*)wmma_ker<chunks_per_warp, warps_per_block, log_had_size>;
        cudaFuncSetAttribute(func_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    }
    if (num_chunks % (chunks_per_warp * warps_per_block) != 0) {
        #ifndef NOPYBIND
        pybind11::print("chunks not divisible by chunks per block");
        #else
        printf("chunks not divisible by chunks per block\n");
        #endif
        return;
    }
    wmma_ker<chunks_per_warp, warps_per_block, log_had_size><<<dim3(grid_size), dim3(block_size), shared_size>>>((half*) a_mat);
}

int main()
{
    printf("hi\n");

    // 16x4096
    constexpr uint32_t log_had_size = 15;
    uint32_t vector_size = 1 << log_had_size;
    uint32_t cols = 16 * (1 << 20) / vector_size; // 16m elements
    // for size 256, use (2, 1)
    // for size 32k use (8, 16)
    constexpr int chunks_per_warp = 8;
    constexpr int warps_per_block = 16;

    half* ptr = (half*)malloc(vector_size * cols * sizeof(half)); // col major

    for(int i = 0; i < vector_size * cols; i++) ptr[i] = (i % (1 << log_had_size) == i / (1 << log_had_size));
    half* dev_ptr;
    cudaMalloc(&dev_ptr, vector_size * cols * sizeof(half));
    cudaMemcpy(dev_ptr, ptr, vector_size * cols * sizeof(half), cudaMemcpyKind::cudaMemcpyHostToDevice);

    printf("cols: %d\n", cols);
    run_matmul_kernel<chunks_per_warp, warps_per_block, log_had_size>(dev_ptr, (cols * vector_size / 256));
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("error\n");
        printf("CUDA Error %d: %s", status, cudaGetErrorString(status));
        printf("\n");
        return 1;
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        printf("error\n");
        printf("CUDA Sync Error %d: %s", status, cudaGetErrorString(status));
        printf("\n");
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

#ifndef NOPYBIND

void fast_had_trans(uint64_t a, uint32_t M, uint32_t N, uint32_t log_had_size) {
    void* a_mat = (void*) a;
    uint32_t num_chunks = (M * N + 255) / 256;
    if ((M * N) % 256 != 0) {
        size_t old_size = M * N * sizeof(half), new_size = num_chunks * 256 * sizeof(half);
        cudaMalloc(&a_mat, new_size);
        cudaMemcpy(a_mat, (void*) a, old_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        cudaMemset((half*) a_mat + M * N, 0, new_size - old_size);
    }
    // for size 256, use (2, 1)
    // for size 32k use (8, 16)
    constexpr int chunks_per_warp_small = 1;// 8;
    constexpr int warps_per_block_small = 1;//2;//16;
    constexpr int chunks_per_warp_large = 2;
    constexpr int warps_per_block_large = 1;

    if (M * N <= 256) {
        switch (log_had_size) {
            case 1: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 1>((half*) a_mat, num_chunks); break;
            case 2: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 2>((half*) a_mat, num_chunks); break;
            case 3: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 3>((half*) a_mat, num_chunks); break;
            case 4: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 4>((half*) a_mat, num_chunks); break;
            case 5: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 5>((half*) a_mat, num_chunks); break;
            case 6: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 6>((half*) a_mat, num_chunks); break;
            case 7: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 7>((half*) a_mat, num_chunks); break;
            case 8: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 8>((half*) a_mat, num_chunks); break;
            default:
                pybind11::print("Invalid log_had_size: %d\n", log_had_size);
                return;
        }
    } else {
        switch (log_had_size) {
            case 1: run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 1>((half*) a_mat, num_chunks); break;
            case 2: run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 2>((half*) a_mat, num_chunks); break;
            case 3: run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 3>((half*) a_mat, num_chunks); break;
            case 4: run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 4>((half*) a_mat, num_chunks); break;
            case 5: run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 5>((half*) a_mat, num_chunks); break;
            case 6: run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 6>((half*) a_mat, num_chunks); break;
            case 7: run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 7>((half*) a_mat, num_chunks); break;
            case 8: run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 8>((half*) a_mat, num_chunks); break;
            case 9: run_matmul_kernel<2, 1, 9>((half*) a_mat, num_chunks); break;
            case 10: run_matmul_kernel<2, 2, 10>((half*) a_mat, num_chunks); break;
            case 11: run_matmul_kernel<2, 4, 11>((half*) a_mat, num_chunks); break;
            case 12: run_matmul_kernel<2, 8, 12>((half*) a_mat, num_chunks); break;
            case 13: run_matmul_kernel<2, 16, 13>((half*) a_mat, num_chunks); break;
            case 14: run_matmul_kernel<4, 16, 14>((half*) a_mat, num_chunks); break;
            case 15: run_matmul_kernel<8, 16, 15>((half*) a_mat, num_chunks); break;
            default:
                pybind11::print("Invalid log_had_size: %d\n", log_had_size);
                return;
        }
    }

    auto status = cudaGetLastError();
    auto status1 = cudaDeviceSynchronize();
    if (status != cudaSuccess || status1 != cudaSuccess) {
        if (status != cudaSuccess) pybind11::print("CUDA Error %d: %s", (long)status, cudaGetErrorString(status));
        if (status1 != cudaSuccess) pybind11::print("CUDA Sync Error %d: %s", (long)status1, cudaGetErrorString(status1));
        pybind11::print("\n");
        pybind11::print("log_had_size: \n", log_had_size);
        pybind11::print("num_chunks: \n", num_chunks);
        return;
    }

    if ((uint64_t) a_mat != a) {
        cudaMemcpy((void*) a, a_mat, M * N * sizeof(half), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        cudaFree(a_mat);
    }
}

namespace py = pybind11;
PYBIND11_MODULE(tensor_core_had_async, m) {
    m.doc() = "fast hadamard transform module";
    m.def("fast_had_trans_async", &fast_had_trans, "A function to perform a fast hadamard transform", py::arg("a"), py::arg("M"), py::arg("N"), py::arg("log_had_size"));
}

#endif
