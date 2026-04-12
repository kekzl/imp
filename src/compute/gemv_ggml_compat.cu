// ggml-compatible Q4_K×Q8_1 MoE GEMV kernel for Gemma-4.
// Matches llama.cpp's vec_dot_q4_K_q8_1 accumulation order exactly.
//
// Key difference from imp's dp4a kernel: each thread processes elements from
// MULTIPLE Q4_K sub-blocks, interleaving dp4a with per-Q8_1-block scale
// multiplication. This produces bit-identical FP32 accumulation as llama.

#include "compute/gemv_ggml_compat.h"
#include "core/logging.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// Q4_K constants (matching ggml)
static constexpr int QK4_K = 256;
static constexpr int QK8_1 = 32;

// Warp reduction
template<int warp_sz = 32>
static __device__ __forceinline__ float warp_reduce_sum_f(float val) {
    #pragma unroll
    for (int mask = warp_sz / 2; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// Extract scale/min from Q4_K scales array (matching ggml get_scale_min_k4)
static __device__ __forceinline__ void get_scale_min(
    const uint8_t* sc, int j, uint8_t& sc_val, uint8_t& min_val)
{
    if (j < 4) {
        sc_val  = sc[j] & 63;
        min_val = sc[j + 4] & 63;
    } else {
        sc_val  = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        min_val = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4);
    }
}

// ggml-compatible Q4_K×Q8_1 dot product for 32 elements (one sub-block).
// Matches vec_dot_q4_K_q8_1_impl_vmmq's accumulation pattern.
static __device__ __forceinline__ float ggml_vec_dot_q4k_q8_1_sub(
    const uint8_t* qs_base,  // Q4_K qs pointer (128 bytes from super-block)
    bool use_high,           // low (false) or high (true) nibbles
    const int* q8_qs,        // Q8_1 int8 values packed as int32[8]
    float q8_d,              // Q8_1 block scale
    float d_super,           // Q4_K super-block delta
    float dmin_super,        // Q4_K super-block dmin
    uint8_t sc_val,          // 6-bit sub-block scale
    uint8_t min_val)         // 6-bit sub-block min
{
    int32_t sumi = 0;
    int32_t q8_sum = 0;
    const int ones = 0x01010101;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint32_t qs4;
        memcpy(&qs4, qs_base + j * 4, 4);
        uint32_t nibbles = use_high ? ((qs4 >> 4) & 0x0F0F0F0Fu)
                                    : (qs4 & 0x0F0F0F0Fu);
        int ni;
        memcpy(&ni, &nibbles, 4);
        sumi = __dp4a(ni, q8_qs[j], sumi);
        q8_sum = __dp4a(q8_qs[j], ones, q8_sum);
    }

    return q8_d * (d_super * (float)sc_val * (float)sumi
                 - dmin_super * (float)min_val * (float)q8_sum);
}

// MoE GEMV: y[rows] = x[K] @ W[rows, K]^T for ONE expert
// Each warp computes one output row. Each thread processes K/warp_size
// super-blocks and accumulates via warp reduction.
__global__ void gemv_q4k_ggml_compat_kernel(
    const uint8_t* __restrict__ W,     // Q4_K packed weights [rows, K/256*144]
    const half* __restrict__ x,        // FP16 input [K]
    half* __restrict__ y,              // FP16 output [rows]
    int rows, int K)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int blocks_per_row = K / QK4_K;
    const int row_bytes = blocks_per_row * 144;
    const uint8_t* W_row = W + static_cast<int64_t>(row) * row_bytes;

    // Quantize x to Q8_1 on the fly (per warp, collaborative)
    // Each Q8_1 block covers 32 FP16 values
    extern __shared__ char smem[];
    int q8_blocks = K / QK8_1;

    // Simple per-thread quantization of x to Q8_1 in shared memory
    struct Q8Block { float d; int8_t qs[32]; };
    Q8Block* q8 = reinterpret_cast<Q8Block*>(smem);

    for (int b = tid; b < q8_blocks; b += blockDim.x) {
        float amax = 0.0f;
        float vals[32];
        for (int i = 0; i < 32; i++) {
            vals[i] = __half2float(x[b * 32 + i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        }
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? 127.0f / amax : 0.0f;
        q8[b].d = d;
        for (int i = 0; i < 32; i++) {
            q8[b].qs[i] = (int8_t)roundf(vals[i] * id);
        }
    }
    __syncthreads();

    // Compute dot product: each thread handles a subset of Q4_K super-blocks
    float sum = 0.0f;
    for (int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        const uint8_t* bp = W_row + blk * 144;
        float d_super = __half2float(*reinterpret_cast<const half*>(bp));
        float dmin_super = __half2float(*reinterpret_cast<const half*>(bp + 2));
        const uint8_t* sc = bp + 4;
        const uint8_t* qs = bp + 16;

        // Process 8 sub-blocks of 32 elements each
        for (int sub = 0; sub < 8; sub++) {
            int q8_idx = blk * 8 + sub;  // Q8_1 block index
            uint8_t sc_val, min_val;
            get_scale_min(sc, sub, sc_val, min_val);

            // Load Q8_1 block data
            int q8_qs[8];
            for (int j = 0; j < 8; j++) {
                memcpy(&q8_qs[j], &q8[q8_idx].qs[j * 4], 4);
            }

            const int qs_byte_offset = (sub / 2) * 32;
            const bool use_high = (sub & 1);

            sum += ggml_vec_dot_q4k_q8_1_sub(
                qs + qs_byte_offset, use_high,
                q8_qs, q8[q8_idx].d,
                d_super, dmin_super, sc_val, min_val);
        }
    }

    // Warp reduction
    sum = warp_reduce_sum_f(sum);

    if (tid == 0) {
        y[row] = __float2half(sum);
    }
}

void gemv_q4k_ggml_compat(
    const uint8_t* W, const half* x, half* y,
    int rows, int K, cudaStream_t stream)
{
    int q8_blocks = K / QK8_1;
    size_t smem = q8_blocks * (sizeof(float) + 32);  // Q8Block per block
    int threads = 32;  // one warp per row
    gemv_q4k_ggml_compat_kernel<<<rows, threads, smem, stream>>>(W, x, y, rows, K);
}

} // namespace imp
