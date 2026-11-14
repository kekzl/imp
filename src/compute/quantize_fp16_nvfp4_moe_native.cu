// Device-side MoE prefill helpers for the grouped NVFP4 device-args path: per-expert M from
// the routing offsets, active-expert alpha compaction, SFA offsets/bases and SFA zeroing.

#include "compute/quantize_fp16_nvfp4_moe_native.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cfloat>
#include <cassert>

namespace imp {

// Per-expert token count from offset scan. One thread per expert, single
// block (n_experts <= ~256).
__global__ void moe_compute_M_per_kernel(
    const int32_t* __restrict__ d_offsets,
    int32_t* __restrict__ d_M_per_out,
    int n_experts)
{
    int e = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (e < n_experts)
        d_M_per_out[e] = d_offsets[e + 1] - d_offsets[e];
}

void compute_M_per_from_offsets_device(
    const int32_t* d_expert_offsets,
    int32_t* d_M_per_out,
    int n_experts,
    cudaStream_t stream)
{
    if (n_experts <= 0) return;
    int threads = std::min(n_experts, 256);
    int blocks  = (n_experts + threads - 1) / threads;
    moe_compute_M_per_kernel<<<blocks, threads, 0, stream>>>(
        d_expert_offsets, d_M_per_out, n_experts);
    IMP_CUDA_CHECK_LAUNCH();
}

// Order-preserving stream compaction of d_alpha to entries where d_M_per[e] > 0.
// Single block, Hillis-Steele inclusive prefix sum in shared memory.
// Experts per thread in the single-block scans below: n_experts <= 256 * kScanMaxChunk.
constexpr int kScanMaxChunk = 16;

__global__ void compact_alpha_active_kernel(
    const float*   __restrict__ d_alpha,
    const int32_t* __restrict__ d_M_per,
    float*         __restrict__ d_alpha_compact,
    int32_t*       __restrict__ d_na_out,
    int n_experts)
{
    // Thread t owns experts [t * chunk, (t + 1) * chunk): local count, block scan of the
    // 256 chunk totals, then the chunk's outputs. n_experts up to 256 * kScanMaxChunk
    // (Qwen3.8-Flash-Next has 512; the one-expert-per-thread form silently capped at 256).
    constexpr int NT = 256;
    __shared__ int s_scan[NT];

    const int t = threadIdx.x;
    const int chunk = (n_experts + NT - 1) / NT;
    const int e0 = t * chunk;
    int local = 0;
    for (int i = 0; i < chunk; ++i) {
        const int e = e0 + i;
        if (e < n_experts && d_M_per[e] > 0)
            ++local;
    }
    s_scan[t] = local;
    __syncthreads();

    // Hillis–Steele inclusive prefix sum over the chunk totals.
    for (int off = 1; off < NT; off <<= 1) {
        int v = (t >= off) ? s_scan[t - off] : 0;
        __syncthreads();
        s_scan[t] += v;
        __syncthreads();
    }

    int excl = s_scan[t] - local;
    for (int i = 0; i < chunk; ++i) {
        const int e = e0 + i;
        if (e < n_experts && d_M_per[e] > 0)
            d_alpha_compact[excl++] = d_alpha[e];
    }
    if (t == 0)
        *d_na_out = s_scan[NT - 1];
}

void compact_alpha_active(
    const float* d_alpha,
    const int32_t* d_M_per,
    float* d_alpha_compact,
    int32_t* d_na_out,
    int n_experts,
    cudaStream_t stream)
{
    if (n_experts <= 0) {
        if (d_na_out)
            IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_na_out, 0, sizeof(int32_t), stream));
        return;
    }
    IMP_CHECK(n_experts <= 256 * kScanMaxChunk,
              "compact_alpha_active: n_experts=%d exceeds the single-block scan limit %d", n_experts,
              256 * kScanMaxChunk);
    compact_alpha_active_kernel<<<1, 256, 0, stream>>>(
        d_alpha, d_M_per, d_alpha_compact, d_na_out, n_experts);
    IMP_CUDA_CHECK_LAUNCH();
}

// Exclusive prefix sum of cutlass_nvfp4_sf_size (per-expert SfAtom-padded SFA byte size).
// Single block, Hillis-Steele scan over int64 in shared memory.
// Padding constants must stay in lockstep with kAtomRows/kAtomKElems/kAtomSize in
// src/compute/nvfp4_sfatom_quant.cu (CUTLASS SfAtom = 128 rows x 64 K-elems x 512 bytes).
// Kept as constexpr here to avoid a host-only include from a device translation unit.
namespace {
constexpr int kSfAtomRows   = 128;
constexpr int kSfAtomKElems = 64;
constexpr int kSfAtomSize   = 512;
}  // anonymous

__global__ void compute_sfa_offsets_kernel(const int32_t* __restrict__ d_M_per,
                                           int64_t* __restrict__ d_sfa_offsets_out,
                                           uint8_t** __restrict__ d_sfa_bases_out,
                                           uint8_t* __restrict__ base_sf, int n_experts, int K) {
    // Same chunked layout as compact_alpha_active_kernel: thread t owns experts
    // [t * chunk, (t + 1) * chunk), block scan over the 256 chunk totals.
    constexpr int NT = 256;
    __shared__ int64_t s_scan[NT];

    const int t = threadIdx.x;
    const int n_k_tiles = (K + kSfAtomKElems - 1) / kSfAtomKElems;
    const int chunk = (n_experts + NT - 1) / NT;
    const int e0 = t * chunk;
    auto bytes_of = [&](int e) -> int64_t {
        const int n_row_tiles = (d_M_per[e] + kSfAtomRows - 1) / kSfAtomRows;
        return static_cast<int64_t>(n_row_tiles) * n_k_tiles * kSfAtomSize;
    };
    int64_t local = 0;
    for (int i = 0; i < chunk; ++i) {
        const int e = e0 + i;
        if (e < n_experts)
            local += bytes_of(e);
    }
    s_scan[t] = local;
    __syncthreads();

    // Hillis–Steele inclusive prefix sum over the chunk totals.
    for (int off = 1; off < NT; off <<= 1) {
        int64_t v = (t >= off) ? s_scan[t - off] : 0;
        __syncthreads();
        s_scan[t] += v;
        __syncthreads();
    }

    // Exclusive prefix per expert; trailing total at slot n_experts.
    int64_t excl = s_scan[t] - local;
    for (int i = 0; i < chunk; ++i) {
        const int e = e0 + i;
        if (e < n_experts) {
            d_sfa_offsets_out[e] = excl;
            if (d_sfa_bases_out)
                d_sfa_bases_out[e] = base_sf + excl;
            excl += bytes_of(e);
        }
    }
    if (t == 0)
        d_sfa_offsets_out[n_experts] = s_scan[NT - 1];
}

void compute_sfa_offsets_device(const int32_t* d_M_per, int64_t* d_sfa_offsets_out, int n_experts, int K,
                                cudaStream_t stream, uint8_t** d_sfa_bases_out, void* base_sf) {
    if (n_experts <= 0) {
        if (d_sfa_offsets_out)
            IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_sfa_offsets_out, 0, sizeof(int64_t), stream));
        return;
    }
    IMP_CHECK(n_experts <= 256 * kScanMaxChunk,
              "compute_sfa_offsets: n_experts=%d exceeds the single-block scan limit %d", n_experts,
              256 * kScanMaxChunk);
    compute_sfa_offsets_kernel<<<1, 256, 0, stream>>>(d_M_per, d_sfa_offsets_out, d_sfa_bases_out,
                                                      static_cast<uint8_t*>(base_sf), n_experts, K);
    IMP_CUDA_CHECK_LAUNCH();
}

// Bounded-prefix wipe of the SfAtom staging buffer, using the device-resident exclusive
// prefix sum's terminal slot as the byte count (replaces cudaMemsetAsync(dst,0,max_bytes);
// sparse routing often uses <50% of the worst-case bound).
// Vectorized 16-byte stores via uint4; grid covers max_bytes, out-of-range threads early-exit.
__global__ void bzero_sfa_active_kernel(
    uint8_t*       __restrict__ dst,
    const int64_t* __restrict__ d_sfa_offsets,
    int            n_experts,
    int64_t        max_bytes)
{
    const int64_t total = (n_experts > 0) ? d_sfa_offsets[n_experts] : 0;
    const int64_t bound = (total < max_bytes) ? total : max_bytes;

    const int64_t tid = (blockIdx.x * (int64_t)blockDim.x) + threadIdx.x;
    const int64_t stride  = (int64_t)gridDim.x * blockDim.x;

    // Vectorized 16-byte stores
    uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    const int64_t vec_bound = bound >> 4;  // bytes / 16
    for (int64_t i = tid; i < vec_bound; i += stride) {
        reinterpret_cast<uint4*>(dst)[i] = zero;
    }
    // Tail bytes (< 16) — first thread handles them
    if (tid == 0) {
        for (int64_t i = vec_bound << 4; i < bound; ++i)
            dst[i] = 0;
    }
}

void bzero_sfa_active(
    void* dst,
    const int64_t* d_sfa_offsets,
    int n_experts,
    size_t max_bytes,
    cudaStream_t stream)
{
    if (!dst || max_bytes == 0) return;
    if (!d_sfa_offsets || n_experts <= 0) {
        // No offsets known — fall back to full memset (rare path, e.g. cold init).
        IMP_CUDA_CHECK_LOG(cudaMemsetAsync(dst, 0, max_bytes, stream));
        return;
    }
    // Grid sized for full saturation at max_bytes; each thread does 16 bytes.
    constexpr int kThreads = 256;
    const int64_t vec_max = static_cast<int64_t>(max_bytes >> 4);
    int64_t blocks_needed = (vec_max + kThreads - 1) / kThreads;
    blocks_needed = std::max<int64_t>(blocks_needed, 1);
    blocks_needed = std::min<int64_t>(blocks_needed, 4096);  // cap grid
    bzero_sfa_active_kernel<<<static_cast<int>(blocks_needed), kThreads, 0, stream>>>(
        static_cast<uint8_t*>(dst), d_sfa_offsets, n_experts,
        static_cast<int64_t>(max_bytes));
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
