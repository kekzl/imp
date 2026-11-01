// KV eviction proof harness: TriAttention scoring on the paged cache and the decode-time block-table
// compaction. Design and scope: docs/plans/2026-11-01-kv-eviction.md. Runs only with
// kv_cache.evict_keep_ratio > 0; nothing here is on the default path.

#include "exec/kv_evict.h"
#include "compute/warp_reduce.cuh"
#include "core/logging.h"
#include "quant/turboquant_fp4.cuh"
#include <cuda_fp8.h>
#include <float.h>

namespace imp {

namespace {

constexpr int kQStatsTile = 64;  // query rows per CTA of the statistics kernel
constexpr int kScoreWarps = 8;   // warps per CTA of the score kernel
constexpr int kScoreTile = 256;  // tokens per CTA of the score kernel
constexpr int kMaxGroup = 8;     // query heads per KV head

// Pre-RoPE statistics: one thread per rotary pair, the angle is shared by every head of a row.
__global__ void kv_evict_qstats_kernel(const half* __restrict__ q, const int* __restrict__ positions, int n,
                                       int n_heads, int head_dim, bool neox, float theta, float inv_scaling,
                                       float* __restrict__ stats) {
    extern __shared__ float s_acc[];  // [n_heads][pairs][3]
    const int pairs = head_dim / 2;
    const int f = threadIdx.x;
    for (int i = threadIdx.x; i < n_heads * pairs * 3; i += blockDim.x)
        s_acc[i] = 0.0f;
    __syncthreads();
    const int t0 = blockIdx.x * kQStatsTile;
    const int t1 = min(t0 + kQStatsTile, n);
    if (f < pairs) {
        // Same frequency expression as rope_angle's linear mode (compute/rope.cu).
        float freq = 1.0f / powf(theta, (2.0f * f) / static_cast<float>(2 * pairs));
        freq *= inv_scaling;
        const int i0 = neox ? f : 2 * f;
        const int i1 = neox ? f + pairs : (2 * f) + 1;
        for (int t = t0; t < t1; ++t) {
            double sd, cd;
            sincos(static_cast<double>(positions[t]) * static_cast<double>(freq), &sd, &cd);
            const float c = static_cast<float>(cd), s = static_cast<float>(sd);
            const half* row = q + (static_cast<int64_t>(t) * n_heads * head_dim);
            for (int h = 0; h < n_heads; ++h) {
                const float a2 = __half2float(row[(h * head_dim) + i0]);
                const float b2 = __half2float(row[(h * head_dim) + i1]);
                const float a = (a2 * c) + (b2 * s);  // inverse rotation
                const float b = (b2 * c) - (a2 * s);
                float* acc = s_acc + (static_cast<ptrdiff_t>((h * pairs) + f) * 3);
                acc[0] += a;
                acc[1] += b;
                acc[2] += sqrtf((a * a) + (b * b));
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_heads * pairs * 3; i += blockDim.x)
        atomicAdd(stats + i, s_acc[i]);
    if (threadIdx.x == 0)
        atomicAdd(stats + (static_cast<ptrdiff_t>(n_heads) * pairs * 3), static_cast<float>(t1 - t0));
}

__device__ __forceinline__ unsigned enc_ordered(float v) {
    const unsigned b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float dec_ordered(unsigned e) {
    return __uint_as_float((e & 0x80000000u) ? (e & 0x7FFFFFFFu) : ~e);
}

struct KeyF16 {
    __device__ __forceinline__ static float get(const KvEvictKeyView& kv, int phys, int slot, int e) {
        const half* blk = reinterpret_cast<const half*>(static_cast<const char*>(kv.k_base) +
                                                        (static_cast<int64_t>(phys) * kv.block_stride));
        return __half2float(blk[(static_cast<int64_t>(slot) * kv.n_kv_heads * kv.head_dim) + e]);
    }
};
// FP8 keys without the per-tensor scale: a per-layer constant cancels in the z-normalization.
struct KeyFp8 {
    __device__ __forceinline__ static float get(const KvEvictKeyView& kv, int phys, int slot, int e) {
        const __nv_fp8_e4m3* blk = reinterpret_cast<const __nv_fp8_e4m3*>(
            static_cast<const char*>(kv.k_base) + (static_cast<int64_t>(phys) * kv.block_stride));
        return static_cast<float>(blk[(static_cast<int64_t>(slot) * kv.n_kv_heads * kv.head_dim) + e]);
    }
};
// NVFP4: two codes per byte (low nibble = even element), UE4M3 scale per 16 elements of a row.
struct KeyNvfp4 {
    __device__ __forceinline__ static float get(const KvEvictKeyView& kv, int phys, int slot, int e) {
        const int row_elems = kv.n_kv_heads * kv.head_dim;
        const uint8_t* blk = static_cast<const uint8_t*>(kv.k_base) +
                             (static_cast<int64_t>(phys) * kv.block_stride);
        const uint8_t byte = blk[(static_cast<int64_t>(slot) * (row_elems / 2)) + (e >> 1)];
        const uint32_t code = (e & 1) ? (byte >> 4) : (byte & 0xF);
        const float mag = kTQFP4DequantLUT[code & 0x7];
        const float v = (code & 0x8) ? -mag : mag;
        const uint8_t* sc = kv.scale_base + (static_cast<int64_t>(phys) * kv.scale_block_stride);
        __nv_fp8_e4m3 sbits;
        const uint8_t raw = sc[(static_cast<int64_t>(slot) * (row_elems / 16)) + (e >> 4)];
        memcpy(&sbits, &raw, 1);
        return v * static_cast<float>(sbits);
    }
};

// One warp per token of one KV head (grid.y); lane holds PPL rotary pairs f = lane + 32 j.
// Score per query head h of the group: u_h . k + sum_f w_h[f] |k_f| (TriAttention Eq. 10 with
// the future-offset mean folded into u, Eq. 11).
template <typename Key, int PPL>
__global__ void kv_evict_score_kernel(int pass, KvEvictKeyView kv, const float* __restrict__ u,
                                      const float* __restrict__ w, const int* __restrict__ bt, int tok_begin,
                                      int tok_end, int n_heads, double* __restrict__ head_stats,
                                      unsigned* __restrict__ tok_max) {
    const int kvh = blockIdx.y;
    const int group = n_heads / kv.n_kv_heads;
    const int hd = kv.head_dim;
    const int pairs = hd / 2;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // NeoX pairs (f, f + pairs) only: the host gate refuses interleaved RoPE.
    __shared__ double s_sum[kScoreWarps][kMaxGroup][2];
    float sum[kMaxGroup], sumsq[kMaxGroup];
    double mean[kMaxGroup], inv_std[kMaxGroup];
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g) {
        sum[g] = 0.0f;
        sumsq[g] = 0.0f;
        mean[g] = 0.0;
        inv_std[g] = 0.0;
    }
    if (pass == 1) {
        const double n = static_cast<double>(tok_end - tok_begin);
        for (int g = 0; g < group && g < kMaxGroup; ++g) {
            const int h = (kvh * group) + g;
            const double m = head_stats[(h * 2) + 0] / n;
            const double var = (head_stats[(h * 2) + 1] / n) - (m * m);
            mean[g] = m;
            inv_std[g] = 1.0 / sqrt(fmax(var, 1e-12));
        }
    }
    const int t_start = tok_begin + (blockIdx.x * kScoreTile);
    const int t_stop = min(t_start + kScoreTile, tok_end);
    for (int t = t_start + warp; t < t_stop; t += kScoreWarps) {
        const int phys = bt[t / kv.block_size];
        const int slot = t % kv.block_size;
        float k0[PPL], k1[PPL], mag[PPL];
#pragma unroll
        for (int j = 0; j < PPL; ++j) {
            const int f = lane + (32 * j);
            k0[j] = Key::get(kv, phys, slot, (kvh * hd) + f);
            k1[j] = Key::get(kv, phys, slot, (kvh * hd) + f + pairs);
            mag[j] = sqrtf((k0[j] * k0[j]) + (k1[j] * k1[j]));
        }
        float zmax = -FLT_MAX;
        for (int g = 0; g < group && g < kMaxGroup; ++g) {
            const int h = (kvh * group) + g;
            const float* uh = u + (static_cast<int64_t>(h) * hd);
            const float* wh = w + (static_cast<int64_t>(h) * pairs);
            float p = 0.0f;
#pragma unroll
            for (int j = 0; j < PPL; ++j) {
                const int f = lane + (32 * j);
                p += (uh[f] * k0[j]) + (uh[f + pairs] * k1[j]) + (wh[f] * mag[j]);
            }
            const float s = warp_reduce_sum(p);
            if (pass == 0) {
                sum[g] += s;
                sumsq[g] += s * s;
            } else {
                zmax = fmaxf(zmax, static_cast<float>((static_cast<double>(s) - mean[g]) * inv_std[g]));
            }
        }
        if (pass == 1 && lane == 0)
            atomicMax(tok_max + t, enc_ordered(zmax));
    }
    if (pass != 0)
        return;
    if (lane == 0) {
        for (int g = 0; g < kMaxGroup; ++g) {
            s_sum[warp][g][0] = sum[g];
            s_sum[warp][g][1] = sumsq[g];
        }
    }
    __syncthreads();
    if (threadIdx.x < group * 2 && threadIdx.x < kMaxGroup * 2) {
        const int g = threadIdx.x >> 1;
        const int c = threadIdx.x & 1;
        double acc = 0.0;
        for (int wi = 0; wi < kScoreWarps; ++wi)
            acc += s_sum[wi][g][c];
        atomicAdd(head_stats + ((((kvh * group) + g) * 2) + c), acc);
    }
}

__global__ void kv_evict_accumulate_kernel(unsigned* __restrict__ tok_max, float* __restrict__ tok_acc,
                                           int tok_begin, int tok_end) {
    const int t = tok_begin + (blockIdx.x * blockDim.x) + threadIdx.x;
    if (t >= tok_end)
        return;
    const unsigned e = tok_max[t];
    if (e != 0u)
        tok_acc[t] += dec_ordered(e);
    tok_max[t] = 0u;
}

// One warp per row: ballot-compacts the live entries of the row's context blocks.
__global__ void kv_evict_compact_kernel(const int* __restrict__ bt, const int* __restrict__ ctx_lens,
                                        int n_rows, int stride, int block_size, int* __restrict__ out_bt,
                                        int* __restrict__ out_ctx) {
    const int row = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    const int lane = threadIdx.x & 31;
    if (row >= n_rows)
        return;
    const int ctx = ctx_lens[row];
    const int n_blk = ctx > 0 ? min((ctx + block_size - 1) / block_size, stride) : 0;
    const int* src = bt + (static_cast<int64_t>(row) * stride);
    int* dst = out_bt + (static_cast<int64_t>(row) * stride);
    int base = 0;
    for (int i0 = 0; i0 < n_blk; i0 += 32) {
        const int i = i0 + lane;
        const int id = (i < n_blk) ? src[i] : -1;
        const unsigned live = __ballot_sync(0xFFFFFFFFu, id >= 0);
        if (id >= 0)
            dst[base + __popc(live & ((1u << lane) - 1u))] = id;
        base += __popc(live);
    }
    if (lane == 0)
        out_ctx[row] = ctx - ((n_blk - base) * block_size);
}

template <typename Key>
void launch_score(int pass, const KvEvictKeyView& kv, const float* u, const float* w, const int* bt,
                  int tok_begin, int tok_end, int n_heads, double* head_stats, unsigned* tok_max,
                  cudaStream_t stream) {
    const dim3 grid((tok_end - tok_begin + kScoreTile - 1) / kScoreTile, kv.n_kv_heads);
    const int ppl = kv.head_dim / 64;
    if (ppl == 1) {
        kv_evict_score_kernel<Key, 1><<<grid, kScoreWarps * 32, 0, stream>>>(pass, kv, u, w, bt, tok_begin,
                                                                             tok_end, n_heads, head_stats,
                                                                             tok_max);
        IMP_CUDA_CHECK_LAUNCH();
    } else if (ppl == 2) {
        kv_evict_score_kernel<Key, 2><<<grid, kScoreWarps * 32, 0, stream>>>(pass, kv, u, w, bt, tok_begin,
                                                                             tok_end, n_heads, head_stats,
                                                                             tok_max);
        IMP_CUDA_CHECK_LAUNCH();
    } else {
        kv_evict_score_kernel<Key, 4><<<grid, kScoreWarps * 32, 0, stream>>>(pass, kv, u, w, bt, tok_begin,
                                                                             tok_end, n_heads, head_stats,
                                                                             tok_max);
        IMP_CUDA_CHECK_LAUNCH();
    }
}

}  // namespace

bool kv_evict_qstats_accumulate(const half* q, const int* positions, int n, int n_heads, int head_dim,
                                bool neox, float theta, float inv_scaling, float* stats_layer,
                                cudaStream_t stream) {
    const int pairs = head_dim / 2;
    if (n <= 0 || pairs <= 0 || pairs > 256 || n_heads * pairs > 4096)
        return false;
    const size_t smem = static_cast<size_t>(n_heads) * pairs * 3 * sizeof(float);
    const int threads = ((pairs + 31) / 32) * 32;
    kv_evict_qstats_kernel<<<(n + kQStatsTile - 1) / kQStatsTile, threads, smem, stream>>>(
        q, positions, n, n_heads, head_dim, neox, theta, inv_scaling, stats_layer);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

void kv_evict_score_pass(int pass, const KvEvictKeyView& kv, const float* u, const float* w, const int* bt,
                         int tok_begin, int tok_end, int n_heads, double* head_stats, unsigned* tok_max,
                         cudaStream_t stream) {
    if (tok_end <= tok_begin || kv.n_kv_heads <= 0 || n_heads / kv.n_kv_heads > kMaxGroup)
        return;
    if (kv.head_dim != 64 && kv.head_dim != 128 && kv.head_dim != 256)
        return;
    if (kv.dtype == QType::NVFP4)
        launch_score<KeyNvfp4>(pass, kv, u, w, bt, tok_begin, tok_end, n_heads, head_stats, tok_max, stream);
    else if (kv.dtype == QType::FP8_E4M3)
        launch_score<KeyFp8>(pass, kv, u, w, bt, tok_begin, tok_end, n_heads, head_stats, tok_max, stream);
    else
        launch_score<KeyF16>(pass, kv, u, w, bt, tok_begin, tok_end, n_heads, head_stats, tok_max, stream);
}

void kv_evict_accumulate_layer(unsigned* tok_max, float* tok_acc, int tok_begin, int tok_end,
                               cudaStream_t stream) {
    if (tok_end <= tok_begin)
        return;
    kv_evict_accumulate_kernel<<<(tok_end - tok_begin + 255) / 256, 256, 0, stream>>>(tok_max, tok_acc,
                                                                                      tok_begin, tok_end);
    IMP_CUDA_CHECK_LAUNCH();
}

void kv_evict_compact_tables(const int* bt, const int* ctx_lens, int n_rows, int stride, int block_size,
                             int* out_bt, int* out_ctx, cudaStream_t stream) {
    if (n_rows <= 0)
        return;
    constexpr int kRowsPerCta = 4;
    kv_evict_compact_kernel<<<(n_rows + kRowsPerCta - 1) / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
        bt, ctx_lens, n_rows, stride, block_size, out_bt, out_ctx);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
