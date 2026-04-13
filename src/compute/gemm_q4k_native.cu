// Native Q4_K_M GEMM: compute matrix multiply directly on Q4_K_M quantized
// weights without FP16 intermediate. Each weight element stays in FP32 during
// the dot product, matching llama.cpp's precision characteristics.
//
// Used for Gemma-4 MoE to eliminate the FP16 rounding that causes progressive
// routing drift over 30 layers with 128 experts.

#include "compute/gemm_q4k_native.h"
#include "core/logging.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

// Q4_K block format: 144 bytes per 256 elements
// [2B d (FP16)] [2B dmin (FP16)] [12B scales] [128B quants]
static constexpr int Q4K_BLOCK_SIZE = 256;
static constexpr int Q4K_BLOCK_BYTES = 144;

// ---------------------------------------------------------------------------
// Q4_K native GEMV kernel: y[N] = x[K] @ W[N,K]^T
// One output element per thread block. K is reduced within the block.
// x is FP16, W is Q4_K_M raw bytes, y is FP16. All computation in FP32.
// ---------------------------------------------------------------------------
__global__ void gemv_q4k_native_kernel(
    const half* __restrict__ x,          // [K] input vector (FP16)
    const uint8_t* __restrict__ w_raw,   // [N, K/256 * 144] Q4_K packed weights
    half* __restrict__ y,                // [N] output vector (FP16)
    int N, int K)
{
    const int n = blockIdx.x;  // output element index
    if (n >= N) return;

    const int tid = threadIdx.x;
    const int blocks_per_row = K / Q4K_BLOCK_SIZE;
    const uint8_t* row_ptr = w_raw + static_cast<int64_t>(n) * blocks_per_row * Q4K_BLOCK_BYTES;

    float sum = 0.0f;

    // Each thread handles a subset of K elements
    for (int blk = tid; blk < blocks_per_row; blk += blockDim.x) {
        const uint8_t* bp = row_ptr + blk * Q4K_BLOCK_BYTES;
        float d    = __half2float(*reinterpret_cast<const half*>(bp));
        float dmin = __half2float(*reinterpret_cast<const half*>(bp + 2));
        const uint8_t* sc = bp + 4;
        const uint8_t* qs = bp + 16;

        // Process all 256 elements in this block
        for (int sub = 0; sub < 8; sub++) {
            // Unpack 6-bit scale and min
            uint8_t sc_val, min_val;
            if (sub < 4) {
                sc_val  = sc[sub] & 63;
                min_val = sc[sub + 4] & 63;
            } else {
                sc_val  = (sc[sub + 4] & 0xF) | ((sc[sub - 4] >> 6) << 4);
                min_val = (sc[sub + 4] >> 4)   | ((sc[sub]     >> 6) << 4);
            }

            float d_sc  = d * static_cast<float>(sc_val);
            float dm_mn = dmin * static_cast<float>(min_val);

            int base = blk * Q4K_BLOCK_SIZE + sub * 32;
            for (int j = 0; j < 32; j++) {
                int i = sub * 32 + j;
                int qs_byte = (i / 64) * 32 + (i % 32);
                int use_high = (i / 32) & 1;
                uint8_t packed = qs[qs_byte];
                int q4 = use_high ? ((packed >> 4) & 0xF) : (packed & 0xF);

                float w_val = d_sc * static_cast<float>(q4) - dm_mn;
                float x_val = __half2float(x[base + j]);
                sum += w_val * x_val;
            }
        }
    }

    // Block reduce
    __shared__ float smem[256];
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        y[n] = __float2half(smem[0]);
    }
}

// ---------------------------------------------------------------------------
// Q4_K native batched GEMV: for each token in [M, K], compute [M, N] output
// by iterating tokens and calling GEMV per token.
// For MoE: called per active expert.
// ---------------------------------------------------------------------------
void gemv_q4k_native(
    const half* x,             // [M, K] input activations (FP16)
    const uint8_t* w_raw,      // [N, K_raw_bytes] Q4_K packed weights
    half* y,                   // [M, N] output (FP16)
    int M, int N, int K,
    cudaStream_t stream)
{
    int threads = 256;
    for (int m = 0; m < M; m++) {
        gemv_q4k_native_kernel<<<N, threads, 0, stream>>>(
            x + static_cast<int64_t>(m) * K,
            w_raw,
            y + static_cast<int64_t>(m) * N,
            N, K);
    }
}

} // namespace imp
