// Qwen4Exp QSA indexer kernels. Contract: qsa_indexer.h.
#include "compute/qsa_indexer.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>

namespace imp {

namespace {

constexpr int D = kQsaDim;
constexpr int kSelThreads = 1024;  // block_rank assumes 32 warps
constexpr int kScoreWarps = 8;

__device__ __forceinline__ float block_sum_128(float v, float* red) {
    // 128 threads = 4 warps.
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = v;
    __syncthreads();
    const float s = red[0] + red[1] + red[2] + red[3];
    __syncthreads();
    return s;
}

// (1+w) RMSNorm over the 128 values of a head held in smem `y` (fp32), then NeoX RoPE on the
// first rope_dim dims at `pos`; both rounding points match the reference (bf16 there, fp16
// here): norm output rounded before the rotation, rotation output rounded on store.
__device__ __forceinline__ float norm_rope_128(float* y, float* red, const half* w, int pos,
                                               const QsaGeom& g) {
    const int d = threadIdx.x;
    const float ss = block_sum_128(y[d] * y[d], red);
    const float inv = rsqrtf(ss / D + g.eps);
    const float yn = __half2float(__float2half(y[d] * inv * (1.0f + __half2float(w[d]))));
    __syncthreads();
    y[d] = yn;
    __syncthreads();
    float out = yn;
    if (d < g.rope_dim) {
        const int half_dim = g.rope_dim / 2;
        const int i = (d < half_dim) ? d : d - half_dim;
        const float inv_freq = powf(g.theta, -2.0f * static_cast<float>(i) / g.rope_dim);
        const float ang = static_cast<float>(pos) * inv_freq;
        const float c = cosf(ang), s = sinf(ang);
        out = (d < half_dim) ? yn * c - y[d + half_dim] * s : yn * c + y[d - half_dim] * s;
    }
    return out;
}

// grid (rows, n_heads), block 128.
__global__ void qsa_prep_queries_kernel(const half* __restrict__ qk, const int* __restrict__ positions,
                                        const half* __restrict__ w_q, half* __restrict__ q_out,
                                        half* __restrict__ raw_keys, QsaGeom g) {
    __shared__ float y[D];
    __shared__ float red[4];
    const int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
    const int stride = (g.n_heads + 1) * D;
    const int pos = positions[row];
    if (head == 0)
        raw_keys[static_cast<size_t>(pos) * D + d] = qk[static_cast<size_t>(row) * stride + g.n_heads * D + d];
    y[d] = __half2float(qk[static_cast<size_t>(row) * stride + head * D + d]);
    __syncthreads();
    const float out = norm_rope_128(y, red, w_q, pos, g);
    q_out[(static_cast<size_t>(row) * g.n_heads + head) * D + d] = __float2half(out);
}

// grid nb, block 128. positions != nullptr: decode, the last complete block before
// positions[0] (one launch, blockIdx.x == 0).
__global__ void qsa_pool_blocks_kernel(const half* __restrict__ raw_keys, const half* __restrict__ w_k,
                                       half* __restrict__ block_keys, int b0,
                                       const int* __restrict__ positions, QsaGeom g) {
    __shared__ float y[D];
    __shared__ float red[4];
    int b;
    if (positions) {
        b = (positions[0] + 1) / g.ratio - 1;
        if (b < 0)
            return;
    } else {
        b = b0 + blockIdx.x;
    }
    const int d = threadIdx.x;
    float acc = 0.0f;
    for (int j = 0; j < g.ratio; ++j)
        acc += __half2float(raw_keys[(static_cast<size_t>(b) * g.ratio + j) * D + d]);
    y[d] = __half2float(__float2half(acc / g.ratio));
    __syncthreads();
    const float out = norm_rope_128(y, red, w_k, b * g.ratio, g);
    block_keys[static_cast<size_t>(b) * D + d] = __float2half(out);
}

// grid (x, rows), block kScoreWarps * 32: one warp per block key (256 B, coalesced),
// grid-stride over the row's complete blocks. score = sum_h relu(q_h . blk) / sqrt(D).
__global__ void __launch_bounds__(kScoreWarps * 32) qsa_score_kernel(const half* __restrict__ q,
                                                                     const int* __restrict__ positions,
                                                                     const half* __restrict__ block_keys,
                                                                     float* __restrict__ scores,
                                                                     int max_blocks, QsaGeom g) {
    const int r = blockIdx.y, lane = threadIdx.x & 31;
    const int nb = (positions[r] + 1) / g.ratio;
    float qr[4][4];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const uint2 u = h < g.n_heads ? *reinterpret_cast<const uint2*>(
                                            q + (static_cast<size_t>(r) * g.n_heads + h) * D + lane * 4)
                                      : make_uint2(0u, 0u);
        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 b = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        qr[h][0] = a.x, qr[h][1] = a.y, qr[h][2] = b.x, qr[h][3] = b.y;
    }
    const float inv_sqrt = rsqrtf(static_cast<float>(D));
    float* my_scores = scores + static_cast<size_t>(r) * max_blocks;
    const int stride = gridDim.x * kScoreWarps;
    for (int b = blockIdx.x * kScoreWarps + (threadIdx.x >> 5); b < nb; b += stride) {
        const uint2 u = *reinterpret_cast<const uint2*>(block_keys + static_cast<size_t>(b) * D + lane * 4);
        const float2 k0 = __half22float2(*reinterpret_cast<const half2*>(&u.x));
        const float2 k1 = __half22float2(*reinterpret_cast<const half2*>(&u.y));
        float dot[4];
#pragma unroll
        for (int h = 0; h < 4; ++h)
            dot[h] = qr[h][0] * k0.x + qr[h][1] * k0.y + qr[h][2] * k1.x + qr[h][3] * k1.y;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (int h = 0; h < 4; ++h)
                dot[h] += __shfl_xor_sync(0xffffffffu, dot[h], o);
        if (lane == 0) {
            float s = 0.0f;
            for (int h = 0; h < g.n_heads; ++h)
                s += fmaxf(dot[h], 0.0f);
            my_scores[b] = s * inv_sqrt;
        }
    }
}

// Exclusive rank of `pred` over the CTA (kSelThreads = 32 warps) and the CTA total, via
// warp ballots. `wsum` holds 33 ints; a caller reuses it only after a barrier.
__device__ __forceinline__ int block_rank(bool pred, int* wsum, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const unsigned m = __ballot_sync(0xffffffffu, pred);
    if (lane == 0)
        wsum[w] = __popc(m);
    __syncthreads();
    if (w == 0) {
        const int v = wsum[lane];
        int incl = v;
        for (int o = 1; o < 32; o <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o)
                incl += n;
        }
        wsum[lane] = incl - v;
        if (lane == 31)
            wsum[32] = incl;
    }
    __syncthreads();
    total = wsum[32];
    return wsum[w] + __popc(m & ((1u << lane) - 1u));
}

// One CTA per query row over the scores of qsa_score_kernel. Radix-selects the k-th largest
// score (exact float key), then compacts the selected blocks in ascending order, ties by
// lowest index.
__global__ void __launch_bounds__(kSelThreads) qsa_select_kernel(const int* __restrict__ positions,
                                                                 const float* __restrict__ scores,
                                                                 int max_blocks,
                                                                 int32_t* __restrict__ sel_tokens,
                                                                 int32_t* __restrict__ sel_count, QsaGeom g) {
    __shared__ int hist[256];
    __shared__ int rank_eq[33], rank_take[33];
    __shared__ int s_bcast[2];
    const int r = blockIdx.x, t = threadIdx.x, lane = t & 31;
    const int p = positions[r];
    const int nb = (p + 1) / g.ratio;
    const int topk = g.budget / g.ratio;
    const int k = nb < topk ? nb : topk;
    const int cap = g.budget + g.ratio - 1;
    const float* my_scores = scores + static_cast<size_t>(r) * max_blocks;
    int32_t* out = sel_tokens + static_cast<size_t>(r) * cap;

    // Threshold key T (k-th largest) and how many ties at T to take, ascending.
    uint32_t T = 0;
    int need_eq = nb;  // nb <= k: everything is selected (keys >= 0 == T)
    if (nb > k) {
        uint32_t prefix = 0, mask_hi = 0;
        int remaining = k;
        for (int shift = 24; shift >= 0; shift -= 8) {
            if (t < 256)
                hist[t] = 0;
            __syncthreads();
            for (int b = t; b < nb; b += kSelThreads) {
                const uint32_t key = __float_as_uint(my_scores[b]);
                if ((key & mask_hi) == prefix)
                    atomicAdd(&hist[(key >> shift) & 255], 1);
            }
            __syncthreads();
            // Warp 0: lane l owns digits 255 - 8l .. 248 - 8l; the lane whose running count
            // (from the top) first reaches `remaining` walks its 8 digits.
            if (t < 32) {
                int c[8], sum = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    c[j] = hist[255 - 8 * lane - j];
                    sum += c[j];
                }
                int incl = sum;
                for (int o = 1; o < 32; o <<= 1) {
                    const int n = __shfl_up_sync(0xffffffffu, incl, o);
                    if (lane >= o)
                        incl += n;
                }
                const unsigned hit = __ballot_sync(0xffffffffu, incl >= remaining);
                if (lane == __ffs(hit) - 1) {
                    int cum = incl - sum, digit = 0;
                    for (int j = 0; j < 8; ++j) {
                        if (cum + c[j] >= remaining) {
                            digit = 255 - 8 * lane - j;
                            break;
                        }
                        cum += c[j];
                    }
                    s_bcast[0] = digit;
                    s_bcast[1] = cum;
                }
            }
            __syncthreads();
            const int digit = s_bcast[0];
            remaining -= s_bcast[1];
            prefix |= static_cast<uint32_t>(digit) << shift;
            mask_hi |= 255u << shift;
            __syncthreads();
        }
        T = prefix;
        need_eq = remaining;  // ties at T still needed after every key > T
    }

    // Compaction in ascending block order.
    int out_blocks = 0, eq_seen = 0;
    for (int base = 0; base < nb; base += kSelThreads) {
        const int b = base + t;
        int is_gt = 0, is_eq = 0;
        if (b < nb) {
            const uint32_t key = __float_as_uint(my_scores[b]);
            is_gt = (nb <= k) || (key > T);
            is_eq = (nb > k) && (key == T);
        }
        int eq_total, take_total;
        const int eq_excl = block_rank(is_eq, rank_eq, eq_total);
        const int take = is_gt || (is_eq && (eq_seen + eq_excl) < need_eq);
        const int take_excl = block_rank(take, rank_take, take_total);
        if (take) {
            const int slot = out_blocks + take_excl;
            for (int j = 0; j < g.ratio; ++j)
                out[slot * g.ratio + j] = b * g.ratio + j;
        }
        out_blocks += take_total;
        eq_seen += eq_total;
        __syncthreads();
    }
    const int tail0 = nb * g.ratio;
    const int n_tail = p + 1 - tail0;
    for (int i = t; i < n_tail; i += kSelThreads)
        out[out_blocks * g.ratio + i] = tail0 + i;
    if (t == 0)
        sel_count[r] = out_blocks * g.ratio + n_tail;
}

// grid (rows, ceil(cap * vec_per_token / 256)), block 256; one int4 (8 halfs) per thread.
__global__ void qsa_gather_kv_kernel(const int4* __restrict__ k_cache, const int4* __restrict__ v_cache,
                                     const int* __restrict__ bt, int block_size, int vec_per_token,
                                     const int32_t* __restrict__ sel_tokens,
                                     const int32_t* __restrict__ sel_count, int cap,
                                     int4* __restrict__ k_scratch, int4* __restrict__ v_scratch,
                                     int blocks_per_row, int32_t* __restrict__ scratch_ctx) {
    const int r = blockIdx.x;
    const int count = sel_count[r];
    if (blockIdx.y == 0 && threadIdx.x == 0)
        scratch_ctx[r] = count;
    const int idx = blockIdx.y * blockDim.x + threadIdx.x;
    const int j = idx / vec_per_token, v = idx - j * vec_per_token;
    if (j >= count)
        return;
    const int tok = sel_tokens[static_cast<size_t>(r) * cap + j];
    const size_t src = (static_cast<size_t>(bt[tok / block_size]) * block_size + tok % block_size) *
                           vec_per_token + v;
    const size_t dst = (static_cast<size_t>(r * blocks_per_row + j / block_size) * block_size +
                        j % block_size) * vec_per_token + v;
    k_scratch[dst] = k_cache[src];
    v_scratch[dst] = v_cache[src];
}

__global__ void qsa_init_scratch_bt_kernel(int32_t* bt, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        bt[i] = i;
}

}  // namespace

void qsa_prep_queries(const half* qk, const int* positions, const half* w_q, half* q_out,
                      half* raw_keys, int rows, const QsaGeom& g, cudaStream_t stream) {
    if (rows <= 0)
        return;
    qsa_prep_queries_kernel<<<dim3(rows, g.n_heads), D, 0, stream>>>(qk, positions, w_q, q_out, raw_keys, g);
    IMP_CUDA_CHECK_LAUNCH();
}

void qsa_pool_blocks(const half* raw_keys, const half* w_k, half* block_keys, int b0, int nb,
                     const int* positions, const QsaGeom& g, cudaStream_t stream) {
    if (positions)
        nb = 1;
    if (nb <= 0)
        return;
    qsa_pool_blocks_kernel<<<nb, D, 0, stream>>>(raw_keys, w_k, block_keys, b0, positions, g);
    IMP_CUDA_CHECK_LAUNCH();
}

void qsa_select(const half* q, const int* positions, const half* block_keys, float* scores,
                int max_blocks, int32_t* sel_tokens, int32_t* sel_count, int rows,
                const QsaGeom& g, cudaStream_t stream) {
    if (rows <= 0)
        return;
    IMP_CHECK(g.n_heads <= 4, "qsa_select: %d indexer heads, kernel holds 4", g.n_heads);
    static const int n_sms = [] {
        int dev = 0, n = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 1;
    }();
    // Two waves of score CTAs over all rows, never more CTAs per row than blocks to score.
    const int gx = std::max(1, std::min((max_blocks + kScoreWarps - 1) / kScoreWarps, 2 * n_sms / rows));
    qsa_score_kernel<<<dim3(gx, rows), kScoreWarps * 32, 0, stream>>>(q, positions, block_keys, scores,
                                                                      max_blocks, g);
    IMP_CUDA_CHECK_LAUNCH();
    qsa_select_kernel<<<rows, kSelThreads, 0, stream>>>(positions, scores, max_blocks, sel_tokens, sel_count,
                                                        g);
    IMP_CUDA_CHECK_LAUNCH();
}

void qsa_gather_kv(const half* k_cache, const half* v_cache, const int* bt, int block_size,
                   int n_kv, int hd, const int32_t* sel_tokens, const int32_t* sel_count,
                   int cap, half* k_scratch, half* v_scratch, int blocks_per_row,
                   int32_t* scratch_ctx, int rows, cudaStream_t stream) {
    if (rows <= 0)
        return;
    const int vec_per_token = n_kv * hd / 8;
    IMP_CHECK((n_kv * hd) % 8 == 0, "qsa_gather_kv: n_kv * hd = %d not a multiple of 8", n_kv * hd);
    const int total = cap * vec_per_token;
    const dim3 grid(rows, (total + 255) / 256);
    qsa_gather_kv_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const int4*>(k_cache), reinterpret_cast<const int4*>(v_cache), bt, block_size,
        vec_per_token, sel_tokens, sel_count, cap, reinterpret_cast<int4*>(k_scratch),
        reinterpret_cast<int4*>(v_scratch), blocks_per_row, scratch_ctx);
    IMP_CUDA_CHECK_LAUNCH();
}

void qsa_init_scratch_bt(int32_t* scratch_bt, int rows, int blocks_per_row, cudaStream_t stream) {
    const int n = rows * blocks_per_row;
    qsa_init_scratch_bt_kernel<<<(n + 255) / 256, 256, 0, stream>>>(scratch_bt, n);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
