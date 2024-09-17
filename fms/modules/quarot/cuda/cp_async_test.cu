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

constexpr int launch_configs_big[7][3] = {
    // default
    {2, 1, 24},
    {2, 2, 16}, 
    {2, 4, 8}, 
    {2, 8, 4}, 
    {2, 16, 3},
    {4, 16, 2},
    {8, 16, 1}
    // // extra coalescing
    // {2, 1, 24},
    // {2, 2, 16}, 
    // {2, 4, 8}, 
    // {2, 8, 4}, 
    // {4, 8, 3},
    // {8, 8, 2},
    // {16, 8, 1}
    // // less coalescing
    // {2, 1, 24},
    // {2, 2, 16}, 
    // {2, 4, 8}, 
    // {2, 8, 4}, 
    // {1, 32, 1},
    // {2, 32, 1},
    // {4, 32, 1}
};

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

template<int num_chunks, int warps_per_block, int log_had_size, int blocks_per_sm>
// TODO: check if blocks per SM is correctly calculated
__global__ void __launch_bounds__(32 * warps_per_block, blocks_per_sm)// MIN(MIN((cudaSharedmemCarveoutMaxShared * 1024) / (num_chunks * warps_per_block * 128 * 4), 24), 48 / warps_per_block))//MIN(24, 48 / warps_per_block))
// a is column major, b is row major
wmma_ker(half* __restrict__ a) {
    uint blockid = blockIdx.x * warps_per_block + threadIdx.x / 32;
    uint threadid = threadIdx.x % 32;
    extern __shared__ b32 bfrag_arr[]; // num_chunks * warps_per_block * 128

    b32 b_frag_all[num_chunks][4]; // for all chunks, holds matrix fragment (which takes 4 regs of fp16x2 * 32 threads)

    b32* a_start_ptr = (b32*) (a + blockid * num_chunks * 256); // offset a to where our threadblock starts
    b32* a_ptr = a_start_ptr + threadid * 4;
    b32* b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * 128 + threadid * 4;
    #pragma unroll
    for (int k = 0; k < num_chunks; k++) {
        // loading for the first iteration

        // load directly to registers
        // thread 0 loads  [t0r0, t16r1, t0r2, t16r3]
        // thread 16 loads [t0r1, t16r0, t0r3, t16r2]
        // allows full coalescing, same for t1/t17, t2/t18, etc.
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int reg = ((threadid & 16) == 0) ? j : (j / 2 * 2 + (1 - j % 2));
            int real_thread_id = (reg == 0 || reg == 2) ? threadid : (threadid ^ 16);
            int real_row = real_thread_id % 4;
            int real_col = real_thread_id / 4;
            b_frag_all[k][j] = ((b32*)a)[(real_row + (reg % 2) * 4) + (real_col + (j / 2) * 8) * 8 + (k + blockid * num_chunks) * 128];
        }

        // for t16 swap r0/r1 and r2/r3 to have [t16r0, t0r1, t16r2, t0r3]
        // so registers are in right order, same for t17, t18, etc.
        if ((threadid & 16) != 0) {
            b32 temp = b_frag_all[k][0];
            b_frag_all[k][0] = b_frag_all[k][1];
            b_frag_all[k][1] = temp;

            temp = b_frag_all[k][2];
            b_frag_all[k][2] = b_frag_all[k][3];
            b_frag_all[k][3] = temp;
        }

        // t0 and t16 swap r1 and r3 to have their own data,
        // same for t1/t17, t2/18, etc.
        #pragma unroll
        for (int j = 1; j < 4; j += 2) {
            b_frag_all[k][j] = __shfl_xor_sync(0xFFFFFFFF, b_frag_all[k][j], 16);
        }

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
        int c_log_h = (i == 0) ? MIN(4, log_had_size) : log_had_size % 4;
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
        if constexpr(log_had_size <= 4 || log_had_size % 4 == 0) break;
    }

    // log had size above 8, only used for above 2^8 = 256 size
    constexpr int part8_log_had_size = log_had_size - 8;
    
    // tensor core register layout row/col
    int row = threadid % 4;
    int col = threadid / 4;

    b32* a_chunk_ptr = a_start_ptr; // our first chunk starts at our threadblock start

    #pragma unroll
    for (int l = 0; l < 2; l++) {
        if constexpr(log_had_size <= 8) { // l == 0 guaranteed, redundant simplified version of else body, to help compiler warnings
            b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * 128;
        } else {
            b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * (l == 0 ? 128 : (128 >> part8_log_had_size));
        }

        // second iteration load
        if (l == 1) {
            if constexpr(log_had_size > 8) {
                __syncthreads(); // sync between first and second iterations if above size 256

                if constexpr(log_had_size < 12) {
                    #pragma unroll
                    for (int k = 0; k < num_chunks; k++) {
                        b32* b_frag_ptr2 = b_frag_ptr + k * (128 >> part8_log_had_size);
                        #define b_frag b_frag_all[k] // TODO: inline this, could cause bugs
                        // sizes 512, 1k, and 2k

                        // for 512:
                        //     thread 0 loads  [t0r0, t0r1, t16r2, t16r3]
                        //     thread 16 loads [t0r2, t0r3, t16r0, t16r1]
                        //     same for t1/t17, t2/t18, etc.
                        // for 1k and 2k:
                        //     thread 0 loads [t0r0, t0r1, t1r2, t1r3]
                        //     thread 1 loads [t0r2, t0r3, t1r0, t1r1]
                        //     same for t2/t3, t4/t5, etc.
                        // allows full coalescing for 512 and 1k, 16x coalescing for 2k
                        constexpr int xor_val = log_had_size == 9 ? 16 : 1;

                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            int reg = ((threadid & xor_val) == 0) ? j : (j + 2) % 4;
                            int real_thread_id = reg < 2 ? threadid : (threadid ^ xor_val);
                            int idx = (real_thread_id / 4 * 16) + (real_thread_id % 4 * 2) + (reg / 2 * 8) + (reg % 2);
                            int rowidx = idx % (1 << part8_log_had_size);
                            int colidx = idx >> part8_log_had_size;
                            b_frag[j] = b_frag_ptr2[rowidx * 128 + colidx];
                        }

                        if ((threadid & xor_val) != 0) {
                            b32 temp = b_frag[0];
                            b_frag[0] = b_frag[2];
                            b_frag[2] = temp;

                            temp = b_frag[1];
                            b_frag[1] = b_frag[3];
                            b_frag[3] = temp;
                        }

                        #pragma unroll
                        for (int j = 2; j < 4; j++) {
                            b_frag[j] = __shfl_xor_sync(0xFFFFFFFF, b_frag[j], xor_val);
                        }
                    }
                } else {
                    // sizes 4k and above

                    // a + threadblock offset + warp offset
                    // can then index into all chunks owned by this warp
                    b32* store = bfrag_arr + (128 >> part8_log_had_size) * (num_chunks * (blockid % warps_per_block));

                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        #pragma unroll
                        for (int k = 0; k < num_chunks; k++) {
                            // here, j represents register, and k represents 8-offset/chunk
                            int real_chunk_num = (num_chunks - (threadid % num_chunks) + k) % num_chunks; // chunk at which you have target thread #'s data
                            
                            int real_thread_id = (threadid / num_chunks) * num_chunks + k; // target thread #
                            int chunk_idx = 128 * real_chunk_num; // index due to fetching from another chunk (chunk in which this thread has the target thread's original data)
                            int thread_group_idx = (real_thread_id / 4) * 16; // index due to fetching from another group of num_chunk threads (since shuffle is between num_chunk threads)
                            int thread_idx = (real_thread_id % 4) * 2; // index due to original thread's position within the group of num_chunk threads
                            int reg_idx = (j / 2) * 8 + (j % 2); // index due to target register
                            int idx = chunk_idx + thread_group_idx + thread_idx + reg_idx; // final index

                            // fix idx for majorness
                            int rowidx = idx % (1 << part8_log_had_size);
                            int colidx = idx >> part8_log_had_size;

                            // store[rowidx * 128 + colidx] = data;
                            b32 data = store[rowidx * 128 + colidx];

                            if (real_chunk_num == 0) b_frag_all[0][j] = data;
                            if constexpr(num_chunks >= 2) {
                                if (real_chunk_num == 1) b_frag_all[1][j] = data;
                            }
                            if constexpr(num_chunks >= 4) {
                                if (real_chunk_num == 2) b_frag_all[2][j] = data;
                                if (real_chunk_num == 3) b_frag_all[3][j] = data;
                            }
                            if constexpr(num_chunks >= 8) {
                                if (real_chunk_num == 4) b_frag_all[4][j] = data;
                                if (real_chunk_num == 5) b_frag_all[5][j] = data;
                                if (real_chunk_num == 6) b_frag_all[6][j] = data;
                                if (real_chunk_num == 7) b_frag_all[7][j] = data;
                            }
                            if constexpr(num_chunks >= 16) {
                                if (real_chunk_num == 8) b_frag_all[8][j] = data;
                                if (real_chunk_num == 9) b_frag_all[9][j] = data;
                                if (real_chunk_num == 10) b_frag_all[10][j] = data;
                                if (real_chunk_num == 11) b_frag_all[11][j] = data;
                                if (real_chunk_num == 12) b_frag_all[12][j] = data;
                                if (real_chunk_num == 13) b_frag_all[13][j] = data;
                                if (real_chunk_num == 14) b_frag_all[14][j] = data;
                                if (real_chunk_num == 15) b_frag_all[15][j] = data;
                            }
                        }
                    }

                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        #pragma unroll
                        for (int k = 1; k < num_chunks; k++) {
                            int threadid_contig = threadid % num_chunks;
                            int threadid_mul = threadid / num_chunks;
                            int threadid2 = (threadid_contig + num_chunks - k) % num_chunks + threadid_mul * num_chunks; // thread to give your data to
                            b_frag[j] = __shfl_sync(0xFFFFFFFF, b_frag[j], threadid2);
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int k = 0; k < num_chunks; k++) {
            #define b_frag b_frag_all[k] // TODO: inline this, could cause bugs

            if (l == 1) {
                // for second iteration, we load 2 consecutive fp16s (1 b32) per register,
                // but tensor core register layout requires 2 fp16s that are in the
                // same column/consecutive rows to be in the same register, so do the swap
                b32 f0 = ((b_frag[1] & 0xFFFF) << 16) | (b_frag[0] & 0xFFFF);
                b32 f1 = ((b_frag[3] & 0xFFFF) << 16) | (b_frag[2] & 0xFFFF);
                b32 f2 = (b_frag[1] & 0xFFFF0000) | (b_frag[0] >> 16);
                b32 f3 = (b_frag[3] & 0xFFFF0000) | (b_frag[2] >> 16);
                b_frag[0] = f0;
                b_frag[1] = f1;
                b_frag[2] = f2;
                b_frag[3] = f3;
            }

            #pragma unroll
            for(int i = 0, remaining_log_had_size = log_had_size - l * 8; i < 2 && remaining_log_had_size > 0; i++) {
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
            }

            if (l == 1) {
                // invert swap from above for second iteration
                b32 f0 = ((b_frag[2] & 0xFFFF) << 16) | (b_frag[0] & 0xFFFF);
                b32 f1 = (b_frag[2] & 0xFFFF0000) | (b_frag[0] >> 16);
                b32 f2 = ((b_frag[3] & 0xFFFF) << 16) | (b_frag[1] & 0xFFFF);
                b32 f3 = (b_frag[3] & 0xFFFF0000) | (b_frag[1] >> 16);
                b_frag[0] = f0;
                b_frag[1] = f1;
                b_frag[2] = f2;
                b_frag[3] = f3;
            }

            if (l == 0) {
                // inverse of coalesced load for first iteration to store result
                #pragma unroll
                for (int j = 1; j < 4; j += 2) {
                    b_frag[j] = __shfl_xor_sync(0xFFFFFFFF, b_frag[j], 16);
                }

                if ((threadid & 16) != 0) {
                    b32 temp = b_frag[0];
                    b_frag[0] = b_frag[1];
                    b_frag[1] = temp;

                    temp = b_frag[2];
                    b_frag[2] = b_frag[3];
                    b_frag[3] = temp;
                }

                // if only going up to 256 size, store directly back to global memory,
                // otherwise store back to shared memory for next iteration
                b32* store = (log_had_size <= 8) ? a_chunk_ptr : b_frag_ptr;

                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int reg = ((threadid & 16) == 0) ? j : (j / 2 * 2 + (1 - j % 2));
                    int real_thread_id = (reg == 0 || reg == 2) ? threadid : (threadid ^ 16);
                    int real_row = real_thread_id % 4;
                    int real_col = real_thread_id / 4;
                    store[(real_row + (reg % 2) * 4) + (real_col + (reg / 2) * 8) * 8] = b_frag[j];
                }
            } else if constexpr(log_had_size > 8) { // condition is redundant to help compiler warnings
                if (log_had_size < 12) {
                    // inverse of coalesced load for sizes 512, 1k and 2k to store result
                    constexpr int xor_val = log_had_size == 9 ? 16 : 1;
                    #pragma unroll
                    for (int j = 2; j < 4; j++) {
                        b_frag[j] = __shfl_xor_sync(0xFFFFFFFF, b_frag[j], xor_val);
                    }

                    if ((threadid & xor_val) != 0) {
                        b32 temp = b_frag[0];
                        b_frag[0] = b_frag[2];
                        b_frag[2] = temp;

                        temp = b_frag[1];
                        b_frag[1] = b_frag[3];
                        b_frag[3] = temp;
                    }

                    b32* store = (b32*)(a + (blockid / warps_per_block) * (num_chunks * warps_per_block) * 256 + (256 >> part8_log_had_size) * (num_chunks * (blockid % warps_per_block) + k));
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int reg = ((threadid & xor_val) == 0) ? j : (j + 2) % 4;
                        b32 data = b_frag_all[k][j];
                        int real_thread_id = reg < 2 ? threadid : (threadid ^ xor_val);
                        int idx = (real_thread_id / 4 * 16) + (real_thread_id % 4 * 2) + (reg / 2 * 8) + (reg % 2);
                        int rowidx = idx % (1 << part8_log_had_size);
                        int colidx = idx >> part8_log_had_size;
                        store[rowidx * 128 + colidx] = data;
                    }
                }
                // for size 4k and above, wait to process all chunks so a final store can be performed coalesced
            }

            a_chunk_ptr += 128; // (only affects first 256 size) move on to next chunk by skipping 256 elements in fp16 (= 128 in b32)
            if constexpr(log_had_size > 8) {
                b_frag_ptr += (l == 0 ? 128 : (128 >> part8_log_had_size));
            } else { // else is redundant, simplified version of if body, to help compiler warnings
                b_frag_ptr += 128;
            }
        }
        if (log_had_size <= 8)
            break;
    }

    if constexpr(log_had_size >= 12) {
        // for sizes 4k and above, perform final coalesced store after processing all chunks

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int k = 1; k < num_chunks; k++) {
                int threadid_contig = threadid % num_chunks;
                int threadid_mul = threadid / num_chunks;
                int threadid2 = (threadid_contig + k) % num_chunks + threadid_mul * num_chunks; // thread to give your data to
                b_frag[j] = __shfl_sync(0xFFFFFFFF, b_frag[j], threadid2);
            }
        }

        // a + threadblock offset + warp offset
        // can then index into all chunks owned by this warp
        b32* store = bfrag_arr + (128 >> part8_log_had_size) * (num_chunks * (blockid % warps_per_block));

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int k = 0; k < num_chunks; k++) {
                // here, j represents register, and k represents 8-offset/chunk
                int real_chunk_num = (num_chunks - (threadid % num_chunks) + k) % num_chunks; // chunk at which you have target thread #'s data
                // b32 data = b_frag_all[real_chunk_num][j]; // target thread data
                b32 data;
                // if statements compile better
                if (real_chunk_num == 0) data = b_frag_all[0][j];
                if constexpr(num_chunks >= 2) {
                    if (real_chunk_num == 1) data = b_frag_all[1][j];
                }
                if constexpr(num_chunks >= 4) {
                    if (real_chunk_num == 2) data = b_frag_all[2][j];
                    if (real_chunk_num == 3) data = b_frag_all[3][j];
                }
                if constexpr(num_chunks >= 8) {
                    if (real_chunk_num == 4) data = b_frag_all[4][j];
                    if (real_chunk_num == 5) data = b_frag_all[5][j];
                    if (real_chunk_num == 6) data = b_frag_all[6][j];
                    if (real_chunk_num == 7) data = b_frag_all[7][j];
                }
                if constexpr(num_chunks >= 16) {
                    if (real_chunk_num == 8) data = b_frag_all[8][j];
                    if (real_chunk_num == 9) data = b_frag_all[9][j];
                    if (real_chunk_num == 10) data = b_frag_all[10][j];
                    if (real_chunk_num == 11) data = b_frag_all[11][j];
                    if (real_chunk_num == 12) data = b_frag_all[12][j];
                    if (real_chunk_num == 13) data = b_frag_all[13][j];
                    if (real_chunk_num == 14) data = b_frag_all[14][j];
                    if (real_chunk_num == 15) data = b_frag_all[15][j];
                }
                if constexpr(num_chunks >= 32) {
                    if (real_chunk_num == 16) data = b_frag_all[16][j];
                    if (real_chunk_num == 17) data = b_frag_all[17][j];
                    if (real_chunk_num == 18) data = b_frag_all[18][j];
                    if (real_chunk_num == 19) data = b_frag_all[19][j];
                    if (real_chunk_num == 20) data = b_frag_all[20][j];
                    if (real_chunk_num == 21) data = b_frag_all[21][j];
                    if (real_chunk_num == 22) data = b_frag_all[22][j];
                    if (real_chunk_num == 23) data = b_frag_all[23][j];
                    if (real_chunk_num == 24) data = b_frag_all[24][j];
                    if (real_chunk_num == 25) data = b_frag_all[25][j];
                    if (real_chunk_num == 26) data = b_frag_all[26][j];
                    if (real_chunk_num == 27) data = b_frag_all[27][j];
                    if (real_chunk_num == 28) data = b_frag_all[28][j];
                    if (real_chunk_num == 29) data = b_frag_all[29][j];
                    if (real_chunk_num == 30) data = b_frag_all[30][j];
                    if (real_chunk_num == 31) data = b_frag_all[31][j];
                }
                
                int real_thread_id = (threadid / num_chunks) * num_chunks + k; // target thread #
                int chunk_idx = 128 * real_chunk_num; // index due to fetching from another chunk (chunk in which this thread has the target thread's original data)
                int thread_group_idx = (real_thread_id / 4) * 16; // index due to fetching from another group of num_chunk threads (since shuffle is between num_chunk threads)
                int thread_idx = (real_thread_id % 4) * 2; // index due to original thread's position within the group of num_chunk threads
                int reg_idx = (j / 2) * 8 + (j % 2); // index due to target register
                int idx = chunk_idx + thread_group_idx + thread_idx + reg_idx; // final index

                // fix idx for majorness
                int rowidx = idx % (1 << part8_log_had_size);
                int colidx = idx >> part8_log_had_size;

                store[rowidx * 128 + colidx] = data;
            }
        }

        __syncthreads();
        store = ((b32*) a) + (blockid / warps_per_block) * (num_chunks * warps_per_block) * 128;
        // flush smem, simply linearly write to store
        #pragma unroll
        for (int warp_off = 0; warp_off < (num_chunks * warps_per_block * 128); warp_off += 32 * warps_per_block) {
            int total_off = warp_off + threadid + (blockid % warps_per_block) * 32;
            store[total_off] = bfrag_arr[total_off];
        }
    }

}


template <int chunks_per_warp, int warps_per_block, int log_had_size, int blocks_per_sm>
void __forceinline__ run_matmul_kernel (half* a_mat, int num_chunks) {
    int shared_size = chunks_per_warp * warps_per_block * 128 * 4;
    dim3 grid_size = num_chunks / chunks_per_warp / warps_per_block;
    dim3 block_size = 32 * warps_per_block;
    if (shared_size > 48 * 1024) {
        void* func_ptr = (void*)wmma_ker<chunks_per_warp, warps_per_block, log_had_size, blocks_per_sm>;
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
    wmma_ker<chunks_per_warp, warps_per_block, log_had_size, blocks_per_sm><<<dim3(grid_size), dim3(block_size), shared_size>>>((half*) a_mat);
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
    constexpr int blocks_per_sm = 1;

    half* ptr = (half*)malloc(vector_size * cols * sizeof(half)); // col major

    for(int i = 0; i < vector_size * cols; i++) ptr[i] = (i % (1 << log_had_size) == i / (1 << log_had_size));
    half* dev_ptr;
    cudaMalloc(&dev_ptr, vector_size * cols * sizeof(half));
    cudaMemcpy(dev_ptr, ptr, vector_size * cols * sizeof(half), cudaMemcpyKind::cudaMemcpyHostToDevice);

    printf("cols: %d\n", cols);
    run_matmul_kernel<chunks_per_warp, warps_per_block, log_had_size, blocks_per_sm>(dev_ptr, (cols * vector_size / 256));
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
    constexpr int blocks_per_sm_small = 24;
    constexpr int chunks_per_warp_large = 2;
    constexpr int warps_per_block_large = 1;
    constexpr int blocks_per_sm_large = 24;

    if (M * N <= 256) {
        switch (log_had_size) {
            case 1: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 1, blocks_per_sm_small>((half*) a_mat, num_chunks); break;
            case 2: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 2, blocks_per_sm_small>((half*) a_mat, num_chunks); break;
            case 3: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 3, blocks_per_sm_small>((half*) a_mat, num_chunks); break;
            case 4: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 4, blocks_per_sm_small>((half*) a_mat, num_chunks); break;
            case 5: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 5, blocks_per_sm_small>((half*) a_mat, num_chunks); break;
            case 6: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 6, blocks_per_sm_small>((half*) a_mat, num_chunks); break;
            case 7: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 7, blocks_per_sm_small>((half*) a_mat, num_chunks); break;
            case 8: run_matmul_kernel<chunks_per_warp_small, warps_per_block_small, 8, blocks_per_sm_small>((half*) a_mat, num_chunks); break;
            default:
                pybind11::print("Invalid log_had_size: %d\n", log_had_size);
                return;
        }
    } else {
        switch (log_had_size) {
            case 1:  run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 1, blocks_per_sm_large>((half*) a_mat, num_chunks); break;
            case 2:  run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 2, blocks_per_sm_large>((half*) a_mat, num_chunks); break;
            case 3:  run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 3, blocks_per_sm_large>((half*) a_mat, num_chunks); break;
            case 4:  run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 4, blocks_per_sm_large>((half*) a_mat, num_chunks); break;
            case 5:  run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 5, blocks_per_sm_large>((half*) a_mat, num_chunks); break;
            case 6:  run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 6, blocks_per_sm_large>((half*) a_mat, num_chunks); break;
            case 7:  run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 7, blocks_per_sm_large>((half*) a_mat, num_chunks); break;
            case 8:  run_matmul_kernel<chunks_per_warp_large, warps_per_block_large, 8, blocks_per_sm_large>((half*) a_mat, num_chunks); break;
            case 9:  run_matmul_kernel<launch_configs_big[0][0], launch_configs_big[0][1], 9 , launch_configs_big[0][2]>((half*) a_mat, num_chunks); break;
            case 10: run_matmul_kernel<launch_configs_big[1][0], launch_configs_big[1][1], 10, launch_configs_big[1][2]>((half*) a_mat, num_chunks); break;
            case 11: run_matmul_kernel<launch_configs_big[2][0], launch_configs_big[2][1], 11, launch_configs_big[2][2]>((half*) a_mat, num_chunks); break;
            case 12: run_matmul_kernel<launch_configs_big[3][0], launch_configs_big[3][1], 12, launch_configs_big[3][2]>((half*) a_mat, num_chunks); break;
            case 13: run_matmul_kernel<launch_configs_big[4][0], launch_configs_big[4][1], 13, launch_configs_big[4][2]>((half*) a_mat, num_chunks); break;
            case 14: run_matmul_kernel<launch_configs_big[5][0], launch_configs_big[5][1], 14, launch_configs_big[5][2]>((half*) a_mat, num_chunks); break;
            case 15: run_matmul_kernel<launch_configs_big[6][0], launch_configs_big[6][1], 15, launch_configs_big[6][2]>((half*) a_mat, num_chunks); break;
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