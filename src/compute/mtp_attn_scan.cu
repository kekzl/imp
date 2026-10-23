// MTP KV-cache softmax attention (draft path): one CTA per (query row, head), split-K for one row over
// long contexts. Split from mtp_forward.cu.

#include "compute/mtp_forward_internal.h"
#include "core/logging.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace imp {

namespace {

constexpr int kScanWarps = kMtpScanThreads / 32;
constexpr int kScanPasses = kMtpScanMaxHeadDim / 256;
// One query row over more than kScanSplitRows cache rows runs split-K (at most kMtpScanMaxSplits CTAs per
// head).
constexpr int kScanSplitRows = 512;

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

// q.k * scale for cache rows [0, seq_len) into s_scores: one warp per row, lane = 8 contiguous dims.
// Returns this warp's max (identical in every lane).
__device__ __forceinline__ float scan_scores(const __half* __restrict__ q_row, const __half* __restrict__ k_h,
                                             int64_t row_stride, int seq_len, int head_dim, float scale,
                                             float* s_scores) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    uint4 qv[kScanPasses];
#pragma unroll
    for (int p = 0; p < kScanPasses; ++p) {
        const int d0 = (p * 256) + (lane * 8);
        qv[p] = d0 < head_dim ? *reinterpret_cast<const uint4*>(q_row + d0) : make_uint4(0, 0, 0, 0);
    }
    float wmax = -1.0e30f;
    for (int t = warp; t < seq_len; t += kScanWarps) {
        const __half* k_row = k_h + (t * row_stride);
        float acc = 0.0f;
#pragma unroll
        for (int p = 0; p < kScanPasses; ++p) {
            const int d0 = (p * 256) + (lane * 8);
            if (d0 < head_dim) {
                const uint4 kv = __ldg(reinterpret_cast<const uint4*>(k_row + d0));
                const auto* q2 = reinterpret_cast<const __half2*>(&qv[p]);
                const auto* k2 = reinterpret_cast<const __half2*>(&kv);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 a = __half22float2(q2[i]);
                    const float2 b = __half22float2(k2[i]);
                    acc += (a.x * b.x) + (a.y * b.y);
                }
            }
        }
        const float s = warp_sum(acc) * scale;
        if (lane == 0)
            s_scores[t] = s;
        wmax = fmaxf(wmax, s);
    }
    return wmax;
}

// Block-wide max (is_max) or sum of one value per warp; ends with every thread past the s_red reads.
__device__ __forceinline__ float block_reduce(float warp_value, bool is_max, float* s_red) {
    if (threadIdx.x % 32 == 0)
        s_red[threadIdx.x / 32] = warp_value;
    __syncthreads();
    float r = s_red[0];
#pragma unroll
    for (int w = 1; w < kScanWarps; ++w)
        r = is_max ? fmaxf(r, s_red[w]) : r + s_red[w];
    __syncthreads();
    return r;
}

// sum_t p_t * V[t] over cache rows [0, seq_len): warps split the rows, lane = 8 dims, reduced in smem.
// part != nullptr: unnormalized sums to part[2 + d]; else out_row[d] = sum * inv_denom.
__device__ __forceinline__ void scan_pv(const __half* __restrict__ v_h, int64_t row_stride, int seq_len,
                                        int head_dim, const float* s_scores, float inv_denom,
                                        __half* __restrict__ out_row, float* part) {
    __shared__ float s_part[kScanWarps][256];
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    for (int c = 0; c < head_dim; c += 256) {
        const int d0 = c + (lane * 8);
        float acc[8] = {};
        if (d0 < head_dim) {
            for (int t = warp; t < seq_len; t += kScanWarps) {
                const uint4 vv = __ldg(reinterpret_cast<const uint4*>(v_h + (t * row_stride) + d0));
                const auto* v2 = reinterpret_cast<const __half2*>(&vv);
                const float pt = s_scores[t];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 b = __half22float2(v2[i]);
                    const int j = 2 * i;
                    acc[j] += pt * b.x;
                    acc[j + 1] += pt * b.y;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
            s_part[warp][(lane * 8) + i] = acc[i];
        __syncthreads();
        const int d = c + static_cast<int>(threadIdx.x);
        if (threadIdx.x < 256 && d < head_dim) {
            float o = 0.0f;
#pragma unroll
            for (int w = 0; w < kScanWarps; ++w)
                o += s_part[w][threadIdx.x];
            if (part != nullptr)
                part[2 + d] = o;
            else
                out_row[d] = __float2half(o * inv_denom);
        }
        __syncthreads();
    }
}

// One (query row, head) over cache rows [0, seq_len). part != nullptr: split-K partial instead of out_row:
// part[0] max, part[1] sum of exp, part[2 + d] unnormalized P*V. blockDim.x == kMtpScanThreads.
__device__ __forceinline__ void mtp_attn_scan_row(const __half* __restrict__ q_row,
                                                  const __half* __restrict__ k_cache,
                                                  const __half* __restrict__ v_cache,
                                                  __half* __restrict__ out_row, int seq_len, int kv_h,
                                                  int num_kv_heads, int head_dim, float scale,
                                                  float* s_scores, float* part = nullptr) {
    __shared__ float s_red[kScanWarps];
    const int64_t row_stride = static_cast<int64_t>(num_kv_heads) * head_dim;
    const int64_t head_off = static_cast<int64_t>(kv_h) * head_dim;
    const float wmax = scan_scores(q_row, k_cache + head_off, row_stride, seq_len, head_dim, scale, s_scores);
    const float gmax = block_reduce(wmax, true, s_red);
    float lsum = 0.0f;
    for (int t = static_cast<int>(threadIdx.x); t < seq_len; t += kMtpScanThreads) {
        const float e = expf(s_scores[t] - gmax);
        s_scores[t] = e;
        lsum += e;
    }
    const float denom = block_reduce(warp_sum(lsum), false, s_red);
    if (part != nullptr && threadIdx.x == 0) {
        part[0] = gmax;
        part[1] = denom;
    }
    scan_pv(v_cache + head_off, row_stride, seq_len, head_dim, s_scores, denom > 0.0f ? 1.0f / denom : 0.0f,
            out_row, part);
}

// Query row r (grid y) attends cache rows [0, base_ctx + r + 1); GQA: q_head h -> kv_head h*NKV/NH.
__global__ void mtp_attn_kv_scan_kernel(const __half* __restrict__ q_attn, const __half* __restrict__ k_cache,
                                        const __half* __restrict__ v_cache, __half* __restrict__ out,
                                        int base_ctx, int num_heads, int num_kv_heads, int head_dim,
                                        float scale) {
    const int h = static_cast<int>(blockIdx.x);
    const int r = static_cast<int>(blockIdx.y);
    if (h >= num_heads)
        return;
    extern __shared__ float s_scores[];  // [seq_len]
    const int kv_h = h / (num_heads / num_kv_heads);
    const int64_t o = ((static_cast<int64_t>(r) * num_heads) + h) * head_dim;
    mtp_attn_scan_row(q_attn + o, k_cache, v_cache, out + o, base_ctx + r + 1, kv_h, num_kv_heads, head_dim,
                      scale, s_scores);
}

// Split-K for one query row: CTA (h, s) scans cache rows [s * chunk, min(seq_len, (s + 1) * chunk)) into
// part[(h * gridDim.y + s) * (head_dim + 2)].
__global__ void mtp_attn_kv_scan_split_kernel(const __half* __restrict__ q,
                                              const __half* __restrict__ k_cache,
                                              const __half* __restrict__ v_cache, float* __restrict__ part,
                                              int seq_len, int chunk, int num_heads, int num_kv_heads,
                                              int head_dim, float scale) {
    const int h = static_cast<int>(blockIdx.x);
    const int s = static_cast<int>(blockIdx.y);
    extern __shared__ float s_chunk_scores[];  // [chunk]
    const int t0 = s * chunk;
    const int64_t row0 = static_cast<int64_t>(t0) * num_kv_heads * head_dim;
    float* p = part + (((static_cast<int64_t>(h) * gridDim.y) + s) * (head_dim + 2));
    mtp_attn_scan_row(q + (static_cast<int64_t>(h) * head_dim), k_cache + row0, v_cache + row0, nullptr,
                      min(chunk, seq_len - t0), h / (num_heads / num_kv_heads), num_kv_heads, head_dim, scale,
                      s_chunk_scores, p);
}

// out[h, :] from head h's n_split partials: rescale each to the global max, sum, normalize.
__global__ void mtp_attn_kv_combine_kernel(const float* __restrict__ part, __half* __restrict__ out,
                                           int n_split, int head_dim) {
    const int h = static_cast<int>(blockIdx.x);
    const int64_t stride = head_dim + 2;
    const float* p = part + (static_cast<int64_t>(h) * n_split * stride);
    float m = -1.0e30f;
    for (int s = 0; s < n_split; ++s)
        m = fmaxf(m, p[s * stride]);
    float l = 0.0f;
    for (int s = 0; s < n_split; ++s)
        l += p[(s * stride) + 1] * expf(p[s * stride] - m);
    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += static_cast<int>(blockDim.x)) {
        float o = 0.0f;
        for (int s = 0; s < n_split; ++s)
            o += p[(s * stride) + 2 + d] * expf(p[s * stride] - m);
        out[(static_cast<int64_t>(h) * head_dim) + d] = __float2half(o * inv_l);
    }
}

// Row r over [0, pos[r] + 1) of slot slots[r].
__global__ void mtp_attn_kv_scan_rows_kernel(const __half* __restrict__ q_attn,
                                             const __half* __restrict__ k_base,
                                             const __half* __restrict__ v_base, __half* __restrict__ out,
                                             const int* __restrict__ slots, const int* __restrict__ pos,
                                             int64_t slot_elems, int num_heads, int num_kv_heads,
                                             int head_dim, float scale) {
    const int h = static_cast<int>(blockIdx.x);
    const int r = static_cast<int>(blockIdx.y);
    if (h >= num_heads)
        return;
    extern __shared__ float s_row_scores[];  // [pos[r] + 1]
    const int64_t slot = static_cast<int64_t>(slots[r]) * slot_elems;
    const int kv_h = h / (num_heads / num_kv_heads);
    const int64_t o = ((static_cast<int64_t>(r) * num_heads) + h) * head_dim;
    mtp_attn_scan_row(q_attn + o, k_base + slot, v_base + slot, out + o, pos[r] + 1, kv_h, num_kv_heads,
                      head_dim, scale, s_row_scores);
}

}  // namespace

bool mtp_scan_smem_opt_in() {
    static const bool ok = [] {
        int dev = 0;
        int optin = 0;
        cudaFuncAttributes fa{};
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess ||
            cudaFuncGetAttributes(&fa, mtp_attn_kv_scan_kernel) != cudaSuccess)
            return false;
        const int dyn = optin - static_cast<int>(fa.sharedSizeBytes);
        return cudaFuncSetAttribute(mtp_attn_kv_scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    dyn) == cudaSuccess &&
               cudaFuncSetAttribute(mtp_attn_kv_scan_rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    dyn) == cudaSuccess;
    }();
    return ok;
}

bool mtp_attn_scan_supported(int head_dim) {
    if (head_dim % 8 != 0 || head_dim > kMtpScanMaxHeadDim) {
        IMP_LOG_ERROR("MTP attention head_dim %d unsupported (multiple of 8, <= %d)", head_dim,
                      kMtpScanMaxHeadDim);
        return false;
    }
    // Scores of the longest row live in dynamic smem: 16k rows = 64 KiB, above the 48 KiB default.
    if (!mtp_scan_smem_opt_in()) {
        IMP_LOG_ERROR("MTP attention: dynamic shared memory opt-in failed");
        return false;
    }
    return true;
}

void mtp_attn_scan(const void* q, const void* k_cache, const void* v_cache, void* out, int base_ctx,
                   int n_rows, int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream,
                   float* split_part) {
    (void)mtp_scan_smem_opt_in();  // false: a launch above 48 KiB fails at IMP_CUDA_CHECK_LAUNCH
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const int seq_len = base_ctx + 1;
    if (n_rows == 1 && split_part != nullptr && seq_len > kScanSplitRows) {
        const int n_split = std::min(kMtpScanMaxSplits, (seq_len + kScanSplitRows - 1) / kScanSplitRows);
        const int chunk = (seq_len + n_split - 1) / n_split;
        const int n_used = (seq_len + chunk - 1) / chunk;  // every split non-empty
        mtp_attn_kv_scan_split_kernel<<<dim3(num_heads, n_used), kMtpScanThreads, chunk * sizeof(float),
                                        stream>>>(static_cast<const __half*>(q),
                                                  static_cast<const __half*>(k_cache),
                                                  static_cast<const __half*>(v_cache), split_part, seq_len,
                                                  chunk, num_heads, num_kv_heads, head_dim, scale);
        IMP_CUDA_CHECK_LAUNCH();
        mtp_attn_kv_combine_kernel<<<num_heads, 256, 0, stream>>>(split_part, static_cast<__half*>(out),
                                                                  n_used, head_dim);
        IMP_CUDA_CHECK_LAUNCH();
        return;
    }
    const size_t smem = static_cast<size_t>(base_ctx + n_rows) * sizeof(float);
    mtp_attn_kv_scan_kernel<<<dim3(num_heads, n_rows), kMtpScanThreads, smem, stream>>>(
        static_cast<const __half*>(q), static_cast<const __half*>(k_cache),
        static_cast<const __half*>(v_cache), static_cast<__half*>(out), base_ctx, num_heads, num_kv_heads,
        head_dim, scale);
    IMP_CUDA_CHECK_LAUNCH();
}

void mtp_attn_scan_slots(const void* q, const void* k_base, const void* v_base, void* out, const int* d_slots,
                         const int* d_pos, int64_t slot_elems, int max_pos, int n_rows, int num_heads,
                         int num_kv_heads, int head_dim, cudaStream_t stream) {
    (void)mtp_scan_smem_opt_in();
    const size_t smem = static_cast<size_t>(max_pos + 1) * sizeof(float);
    mtp_attn_kv_scan_rows_kernel<<<dim3(num_heads, n_rows), kMtpScanThreads, smem, stream>>>(
        static_cast<const __half*>(q), static_cast<const __half*>(k_base), static_cast<const __half*>(v_base),
        static_cast<__half*>(out), d_slots, d_pos, slot_elems, num_heads, num_kv_heads, head_dim,
        1.0f / sqrtf(static_cast<float>(head_dim)));
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
