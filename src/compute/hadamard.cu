// Block-diagonal Walsh-Hadamard transform (WHT) for FP16 activations.
//
// The normalized WHT distributes outliers across elements, reducing the
// dynamic range per micro-block and improving FP4 quantization accuracy.
// This is the runtime component of the MR-GPTQ / QuTLASS approach:
//   X_rotated[i] = (1/sqrt(N)) * H_N @ X[i]   for each block of N elements
//
// The butterfly decomposition gives O(N log N) ops per block vs O(N^2) for dense matmul.
// For N=128: 7 stages × 128 ops = 896 FMA vs 16384 for dense — 18x fewer ops.
//
// Implementation strategy:
//   - Each warp of 32 threads processes one block of up to 32 elements.
//   - For block_size > 32, multiple warps cooperate via shared memory.
//   - Intra-warp butterfly stages use __shfl_xor_sync (zero latency).

#include "compute/hadamard.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cmath>

namespace imp {

// ---------------------------------------------------------------------------
// Walsh-Hadamard butterfly: swap-and-add/sub for one stage.
// partner = threadIdx ^ stride (XOR pattern).
// a[tid] = a[tid] + a[partner]   if tid < partner
// a[tid] = a[partner] - a[tid]   if tid > partner
// ---------------------------------------------------------------------------

// Warp-level butterfly using shuffle.
__device__ __forceinline__
float warp_butterfly(float val, int stage) {
    int stride = 1 << stage;
    float partner = __shfl_xor_sync(0xFFFFFFFF, val, stride);
    int lane = threadIdx.x & 31;
    // If our lane bit at position `stage` is 0: add. Otherwise: sub.
    return (lane & stride) ? (partner - val) : (val + partner);
}

// ---------------------------------------------------------------------------
// Kernel for block_size <= 32 (single warp, no shared memory)
// Each warp handles one block. Threads beyond block_size are masked.
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void hadamard_warp_kernel(
    const half* __restrict__ input,
    half*       __restrict__ output,
    int M, int K)
{
    static_assert(BLOCK_SIZE <= 32, "Use shared memory kernel for block_size > 32");

    // Each warp processes one block of BLOCK_SIZE elements.
    // Grid: one warp per block. Total blocks = M * (K / BLOCK_SIZE).
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int tid_in_warp = threadIdx.x & 31;

    int total_blocks = M * (K / BLOCK_SIZE);
    if (warp_id >= total_blocks) return;
    if (tid_in_warp >= BLOCK_SIZE) return;

    int row = warp_id / (K / BLOCK_SIZE);
    int blk = warp_id % (K / BLOCK_SIZE);
    int base = row * K + blk * BLOCK_SIZE;

    float val = __half2float(input[base + tid_in_warp]);

    // log2(BLOCK_SIZE) butterfly stages
    constexpr int N_STAGES = (BLOCK_SIZE == 16) ? 4 : 5;
    #pragma unroll
    for (int s = 0; s < N_STAGES; s++) {
        val = warp_butterfly(val, s);
    }

    // Normalize: 1/sqrt(BLOCK_SIZE)
    constexpr float norm = (BLOCK_SIZE == 16) ? 0.25f : // 1/sqrt(16) = 0.25
                           (BLOCK_SIZE == 32) ? 0.176776695f : 0.0f; // 1/sqrt(32)
    val *= norm;

    output[base + tid_in_warp] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Kernel for block_size 64: 2 warps cooperate via shared memory.
// Each CTA (2 warps = 64 threads) handles one block of 64 elements.
// Stages 0-4: warp shuffle. Stage 5: shared memory cross-warp.
// ---------------------------------------------------------------------------
__global__ void hadamard_64_kernel(
    const half* __restrict__ input,
    half*       __restrict__ output,
    int M, int K)
{
    constexpr int BLOCK_SIZE = 64;
    constexpr int LOG2_BS = 6;

    // Each CTA = 64 threads = 2 warps. One CTA per block of 64 elements.
    int block_id = blockIdx.x;
    int tid = threadIdx.x;  // 0..63

    int total_blocks = M * (K / BLOCK_SIZE);
    if (block_id >= total_blocks) return;

    int row = block_id / (K / BLOCK_SIZE);
    int blk = block_id % (K / BLOCK_SIZE);
    int base = row * K + blk * BLOCK_SIZE;

    float val = __half2float(input[base + tid]);

    // Stages 0-4: intra-warp butterfly via shuffle
    int lane = tid & 31;
    #pragma unroll
    for (int s = 0; s < 5; s++) {
        val = warp_butterfly(val, s);
    }

    // Stage 5: cross-warp via shared memory (stride = 32)
    __shared__ float smem[64];
    smem[tid] = val;
    __syncthreads();

    int partner = tid ^ 32;
    float pval = smem[partner];
    val = (tid & 32) ? (pval - val) : (val + pval);

    // Normalize: 1/sqrt(64) = 0.125
    val *= 0.125f;

    output[base + tid] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Kernel for block_size 128: 4 warps cooperate via shared memory.
// Each CTA = 128 threads. Stages 0-4: warp shuffle. Stages 5-6: shared memory.
// ---------------------------------------------------------------------------
__global__ void hadamard_128_kernel(
    const half* __restrict__ input,
    half*       __restrict__ output,
    int M, int K)
{
    constexpr int BLOCK_SIZE = 128;

    int block_id = blockIdx.x;
    int tid = threadIdx.x;  // 0..127

    int total_blocks = M * (K / BLOCK_SIZE);
    if (block_id >= total_blocks) return;

    int row = block_id / (K / BLOCK_SIZE);
    int blk = block_id % (K / BLOCK_SIZE);
    int base = row * K + blk * BLOCK_SIZE;

    float val = __half2float(input[base + tid]);

    // Stages 0-4: intra-warp butterfly via shuffle
    #pragma unroll
    for (int s = 0; s < 5; s++) {
        val = warp_butterfly(val, s);
    }

    __shared__ float smem[128];

    // Stage 5: stride 32 (cross-warp pair 0↔1, 2↔3)
    smem[tid] = val;
    __syncthreads();
    {
        int partner = tid ^ 32;
        float pval = smem[partner];
        val = (tid & 32) ? (pval - val) : (val + pval);
    }
    __syncthreads();

    // Stage 6: stride 64 (cross-warp pair 0↔2, 1↔3)
    smem[tid] = val;
    __syncthreads();
    {
        int partner = tid ^ 64;
        float pval = smem[partner];
        val = (tid & 64) ? (pval - val) : (val + pval);
    }

    // Normalize: 1/sqrt(128)
    val *= 0.0883883476f;  // 1/sqrt(128)

    output[base + tid] = __float2half(val);
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

void hadamard_transform_fp16(const half* input, half* output,
                              int M, int K, int block_size,
                              cudaStream_t stream)
{
    if (K % block_size != 0) {
        IMP_LOG_ERROR("hadamard: K=%d not divisible by block_size=%d", K, block_size);
        return;
    }

    int total_blocks = M * (K / block_size);

    switch (block_size) {
    case 16: {
        // 2 warps per CTA, each warp handles one block.
        int warps_per_cta = 8;  // 256 threads
        int blocks_per_cta = warps_per_cta;
        int grid = (total_blocks + blocks_per_cta - 1) / blocks_per_cta;
        hadamard_warp_kernel<16><<<grid, warps_per_cta * 32, 0, stream>>>(
            input, output, M, K);
        break;
    }
    case 32: {
        int warps_per_cta = 8;
        int blocks_per_cta = warps_per_cta;
        int grid = (total_blocks + blocks_per_cta - 1) / blocks_per_cta;
        hadamard_warp_kernel<32><<<grid, warps_per_cta * 32, 0, stream>>>(
            input, output, M, K);
        break;
    }
    case 64: {
        // One CTA (64 threads) per block.
        hadamard_64_kernel<<<total_blocks, 64, 0, stream>>>(
            input, output, M, K);
        break;
    }
    case 128: {
        // One CTA (128 threads) per block.
        hadamard_128_kernel<<<total_blocks, 128, 0, stream>>>(
            input, output, M, K);
        break;
    }
    default:
        IMP_LOG_ERROR("hadamard: unsupported block_size=%d (must be 16/32/64/128)",
                      block_size);
        break;
    }
}

} // namespace imp
