// Qwen4Exp QSA indexer kernels. Contract: qsa_indexer.h.
#include "compute/qsa_indexer.h"
#include "core/logging.h"

#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

namespace {

constexpr int D = kQsaDim;
constexpr int kSelThreads = 256;

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

// Inclusive scan of one int per thread over the block (kSelThreads), result in `buf[tid]`.
__device__ __forceinline__ int block_scan_incl(int v, int* buf) {
    const int t = threadIdx.x;
    buf[t] = v;
    __syncthreads();
    for (int o = 1; o < kSelThreads; o <<= 1) {
        const int u = (t >= o) ? buf[t - o] : 0;
        __syncthreads();
        buf[t] += u;
        __syncthreads();
    }
    return buf[t];
}

// One CTA per query row. Scores, radix-selects the k-th largest score (exact float key),
// then compacts the selected blocks in ascending order, ties by lowest index.
__global__ void __launch_bounds__(kSelThreads)
qsa_select_kernel(const half* __restrict__ q, const int* __restrict__ positions,
                  const half* __restrict__ block_keys, float* __restrict__ scores, int max_blocks,
                  int32_t* __restrict__ sel_tokens, int32_t* __restrict__ sel_count, QsaGeom g) {
    __shared__ float sq[4 * D];
    __shared__ int hist[256];
    __shared__ int scan_buf[kSelThreads];
    __shared__ int s_bcast[4];
    const int r = blockIdx.x, t = threadIdx.x;
    const int p = positions[r];
    const int nb = (p + 1) / g.ratio;
    const int topk = g.budget / g.ratio;
    const int k = nb < topk ? nb : topk;
    const int cap = g.budget + g.ratio - 1;
    float* my_scores = scores + static_cast<size_t>(r) * max_blocks;
    int32_t* out = sel_tokens + static_cast<size_t>(r) * cap;

    for (int i = t; i < g.n_heads * D; i += kSelThreads)
        sq[i] = __half2float(q[static_cast<size_t>(r) * g.n_heads * D + i]);
    __syncthreads();

    const float inv_sqrt = rsqrtf(static_cast<float>(D));
    for (int b = t; b < nb; b += kSelThreads) {
        const half2* bk = reinterpret_cast<const half2*>(block_keys + static_cast<size_t>(b) * D);
        float s = 0.0f;
        for (int h = 0; h < g.n_heads; ++h) {
            float dot = 0.0f;
            const float* qh = sq + h * D;
            for (int i = 0; i < D / 2; ++i) {
                const float2 kv = __half22float2(bk[i]);
                dot += qh[2 * i] * kv.x + qh[2 * i + 1] * kv.y;
            }
            s += fmaxf(dot, 0.0f);
        }
        my_scores[b] = s * inv_sqrt;
    }
    __syncthreads();

    // Threshold key T (k-th largest) and how many ties at T to take, ascending.
    uint32_t T = 0;
    int need_eq = nb;  // nb <= k: everything is selected (keys >= 0 == T)
    if (nb > k) {
        uint32_t prefix = 0, mask_hi = 0;
        int remaining = k;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = t; i < 256; i += kSelThreads)
                hist[i] = 0;
            __syncthreads();
            for (int b = t; b < nb; b += kSelThreads) {
                const uint32_t key = __float_as_uint(my_scores[b]);
                if ((key & mask_hi) == prefix)
                    atomicAdd(&hist[(key >> shift) & 255], 1);
            }
            __syncthreads();
            if (t == 0) {
                int cum = 0, digit = 0;
                for (int dgt = 255; dgt >= 0; --dgt) {
                    if (cum + hist[dgt] >= remaining) {
                        digit = dgt;
                        break;
                    }
                    cum += hist[dgt];
                }
                s_bcast[0] = digit;
                s_bcast[1] = cum;
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
        const int eq_incl = block_scan_incl(is_eq, scan_buf);
        const int eq_total = scan_buf[kSelThreads - 1];
        const int take = is_gt || (is_eq && (eq_seen + eq_incl - 1) < need_eq);
        __syncthreads();
        const int take_incl = block_scan_incl(take, scan_buf);
        const int take_total = scan_buf[kSelThreads - 1];
        if (take) {
            const int slot = out_blocks + take_incl - 1;
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
    qsa_select_kernel<<<rows, kSelThreads, 0, stream>>>(q, positions, block_keys, scores, max_blocks,
                                                        sel_tokens, sel_count, g);
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
