#include "compute/ffn_sparsity_mask.h"
#include "compute/gemm.h"
#include "core/logging.h"
#include "runtime/pdl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

namespace imp {

namespace {

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

// One CUDA block builds the entire mask for one (gate, up) pair.
// Each thread covers multiple K elements via strided loop; per-q8-block
// amax is accumulated into shared memory via atomicMax on float-bit
// patterns (safe for non-negative floats due to IEEE 754 ordering).
template <int THREADS>
__global__ void build_swiglu_block_mask_kernel(const __half* __restrict__ gate,
                                               const __half* __restrict__ up,
                                               uint32_t* __restrict__ mask, int K, float threshold) {
    extern __shared__ unsigned int smem_amax[];  // [K/32] float bits
    const int n_blocks = K >> 5;                 // K / 32

    for (int i = threadIdx.x; i < n_blocks; i += THREADS) {
        smem_amax[i] = 0u;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < K; i += THREADS) {
        const float g = __half2float(gate[i]);
        const float u = __half2float(up[i]);
        const float s = fabsf(silu_f(g) * u);
        atomicMax(&smem_amax[i >> 5], __float_as_uint(s));
    }
    __syncthreads();

    const int n_words = (n_blocks + 31) >> 5;
    const uint32_t thr_bits = __float_as_uint(threshold);
    for (int w = threadIdx.x; w < n_words; w += THREADS) {
        uint32_t bits = 0u;
        const int base = w * 32;
        const int span = n_blocks - base;
        const int upper = span < 32 ? span : 32;
        for (int j = 0; j < upper; ++j) {
            // float-as-uint comparison is monotonic for non-negative IEEE 754
            if (smem_amax[base + j] >= thr_bits) bits |= (1u << j);
        }
        mask[w] = bits;
    }
}

// Kpar-layout masked Q8_0 GEMV with fused residual. Bit-identical to
// gemv_dp4a_kpar_kernel<Q8_0_Traits, true> when every mask bit is set.
template <bool ADD_RESIDUAL>
__global__ void gemv_q8_0_q8_1_residual_masked_kernel(const uint8_t* __restrict__ W,
                                                      const block_q8_1* __restrict__ q8_1,
                                                      const float* __restrict__ d8,
                                                      const uint32_t* __restrict__ mask,
                                                      __half* y, const __half* residual, int M, int K) {
    constexpr int NWARPS = 4;
    constexpr int kBlockBytes = 34;
    constexpr int kBlockElems = 32;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x;
    if (row >= M) return;

    const int total_q8 = K / kBlockElems;
    const size_t row_bytes = static_cast<size_t>(total_q8) * kBlockBytes;
    const uint8_t* row_w = W + static_cast<size_t>(row) * row_bytes;

    float sum = 0.0f;
    constexpr int STRIDE = NWARPS * 32;
    for (int b = warp_id * 32 + lane; b < total_q8; b += STRIDE) {
        // Mask test — skip block if its bit is 0.
        if (!((mask[b >> 5] >> (b & 31)) & 1u)) continue;

        int xi[8];
        memcpy(xi, q8_1[b].qs, 32);
        const float dq = d8[b];

        const uint8_t* bp = row_w + static_cast<size_t>(b) * kBlockBytes;
        __half d_w_h;
        memcpy(&d_w_h, bp, sizeof(__half));
        const float d_w = __half2float(d_w_h);
        int wi[8];
        memcpy(wi, bp + 2, 32);

        int32_t sumi = 0;
        sumi = __dp4a(wi[0], xi[0], sumi);
        sumi = __dp4a(wi[1], xi[1], sumi);
        sumi = __dp4a(wi[2], xi[2], sumi);
        sumi = __dp4a(wi[3], xi[3], sumi);
        sumi = __dp4a(wi[4], xi[4], sumi);
        sumi = __dp4a(wi[5], xi[5], sumi);
        sumi = __dp4a(wi[6], xi[6], sumi);
        sumi = __dp4a(wi[7], xi[7], sumi);

        sum += d_w * dq * static_cast<float>(sumi);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);

    __shared__ float partial[NWARPS];
    if (lane == 0) partial[warp_id] = sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = partial[0] + partial[1] + partial[2] + partial[3];
        if constexpr (ADD_RESIDUAL) total += __half2float(residual[row]);
        y[row] = __float2half(total);
    }
}

}  // namespace

void build_swiglu_block_mask(const __half* gate, const __half* up, uint32_t* mask, int K,
                             float threshold, cudaStream_t stream) {
    constexpr int THREADS = 256;
    const int n_blocks = K >> 5;
    const size_t smem = static_cast<size_t>(n_blocks) * sizeof(unsigned int);
    build_swiglu_block_mask_kernel<THREADS><<<1, THREADS, smem, stream>>>(gate, up, mask, K, threshold);
    IMP_CUDA_CHECK_LAUNCH();
}

void gemv_q8_0_q8_1_residual_masked(const void* W, const block_q8_1* q8_1, const float* d8,
                                    const uint32_t* mask, __half* y, const __half* residual,
                                    int M, int K, cudaStream_t stream) {
    pdl::launch(gemv_q8_0_q8_1_residual_masked_kernel<true>, dim3(M), dim3(128), size_t(0), stream,
                static_cast<const uint8_t*>(W), q8_1, d8, mask, y, residual, M, K);
}

}  // namespace imp
