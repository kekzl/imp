// KDA chunked prefill scan: the WY form of fla's naive_chunk_kda in FP32 on CUDA cores, chunk 16.
// Grid (n_heads, n_seq, kVSplit), 128 threads: 4 threads per value column, 32 key rows each.
// Per chunk with cumulative log decay G_i (per key channel):
//   L[i][j]   = beta_i * sum_c k_i k_j exp(G_i - G_j)   (j < i)
//   Aqk[i][j] = sum_c q_i k_j exp(G_i - G_j)            (j <= i)
//   T = (I + L)^-1 * diag(beta);  w = T (exp(G) . k);  u = T v
//   v' = u - w S;  o = (q . exp(G)) S + Aqk v';  S = exp(G_C) . S + (k . exp(G_C - G))^T v'
// exp(G_i - G_j) is taken per pair; exp(G) alone stays >= exp(16 * lower_bound) = exp(-80) > FLT_MIN.
// Softplus gate (unbounded): exp(G) may flush to 0, it only multiplies (no division), as in fla.

#include "compute/kda_common.cuh"
#include "core/logging.h"

namespace imp {

using namespace kda_detail;

namespace {

constexpr int kC = 16;                         // chunk rows
constexpr int kChSplit = 4;                    // threads per value column
constexpr int kChRows = kHD / kChSplit;        // key rows per thread
constexpr int kChThreads = kVCols * kChSplit;  // 128 == kHD: one thread per key channel in the scans
constexpr int kChWarps = kChThreads / 32;
static_assert(kChThreads == kHD, "per-channel loops assume one thread per key channel");

struct ChunkSmem {
    float q[kC][kHD];  // q (normalised, scaled), then q . exp(G)
    float k[kC][kHD];  // k (normalised), then k . exp(G_C - G)
    float g[kC][kHD];  // log decay, cumsum G, then w
    float e[kC][kHD];  // exp(G) . k
    float L[kC][kC];
    float T[kC][kC];
    float Aqk[kC][kC];
    float beta[kC];
    float eC[kHD];
};

struct ChunkCtx {
    const KdaScanArgs* a;
    KdaSeq sq;
    int h, HK, c0, lc, tid;
    float a_exp, scale;
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ size_t chunk_row(const ChunkCtx& c, int i) {
    return c.sq.row0 + static_cast<size_t>(c.c0 + i);
}

// 1. Raw q, k, log decay, beta; pad rows are zero (no decay, no update).
__device__ void chunk_load(const ChunkCtx& c, ChunkSmem& sm) {
    const KdaScanArgs& a = *c.a;
    const float* dtb = a.dt_bias + (static_cast<size_t>(c.h) * kHD);
    for (int idx = c.tid; idx < kC * kHD; idx += kChThreads) {
        const int i = idx / kHD;
        const int ch = idx % kHD;
        float qv = 0.0f, kv = 0.0f, gv = 0.0f;
        if (i < c.lc) {
            const size_t row = chunk_row(c, i);
            const float* conv = a.conv_f32 + (row * static_cast<size_t>(a.conv_channels));
            qv = conv[(static_cast<size_t>(c.h) * kHD) + ch];
            kv = conv[c.HK + (static_cast<size_t>(c.h) * kHD) + ch];
            const float fv = __half2float(
                a.f[(row * static_cast<size_t>(c.HK)) + (static_cast<size_t>(c.h) * kHD) + ch]);
            gv = kda_log_decay(fv, c.a_exp, dtb[ch], a.lower_bound, a.softplus_gate);
        }
        sm.q[i][ch] = qv;
        sm.k[i][ch] = kv;
        sm.g[i][ch] = gv;
    }
    if (c.tid < kC) {
        float b = 0.0f;
        if (c.tid < c.lc)
            b = kda_sigmoid(
                __half2float(a.beta[(chunk_row(c, c.tid) * static_cast<size_t>(a.n_heads)) + c.h]));
        sm.beta[c.tid] = b;
    }
}

// 2. L2 norm of the 2 * kC rows (one warp per row), cumsum of the decay per channel.
__device__ void chunk_norm_cumsum(const ChunkCtx& c, ChunkSmem& sm) {
    const int lane = c.tid & 31;
    for (int r = c.tid >> 5; r < 2 * kC; r += kChWarps) {
        float* rowp = r < kC ? sm.q[r] : sm.k[r - kC];
        float ss = 0.0f;
#pragma unroll
        for (int m = 0; m < kHD / 32; ++m)
            ss += rowp[lane + (32 * m)] * rowp[lane + (32 * m)];
        const float inv = rsqrtf(warp_sum(ss) + 1e-6f) * (r < kC ? c.scale : 1.0f);
#pragma unroll
        for (int m = 0; m < kHD / 32; ++m)
            rowp[lane + (32 * m)] *= inv;
    }
    float acc = 0.0f;
    for (int i = 0; i < kC; ++i) {
        acc += sm.g[i][c.tid];
        sm.g[i][c.tid] = acc;
    }
}

// 3. Pair products L (strict lower, row-scaled by beta) and Aqk (lower incl. diagonal).
__device__ void chunk_pairs(const ChunkCtx& c, ChunkSmem& sm) {
    const int lane = c.tid & 31;
    for (int p = c.tid >> 5; p < kC * kC; p += kChWarps) {
        const int i = p / kC;
        const int j = p % kC;
        float kk = 0.0f, qk = 0.0f;
        if (j <= i) {
#pragma unroll
            for (int m = 0; m < kHD / 32; ++m) {
                const int ch = lane + (32 * m);
                const float kj = sm.k[j][ch] * __expf(sm.g[i][ch] - sm.g[j][ch]);
                kk += sm.k[i][ch] * kj;
                qk += sm.q[i][ch] * kj;
            }
            kk = warp_sum(kk);
            qk = warp_sum(qk);
        }
        if (lane == 0) {
            sm.L[i][j] = j < i ? sm.beta[i] * kk : 0.0f;
            sm.Aqk[i][j] = j <= i ? qk : 0.0f;
        }
    }
}

// T = (I + L)^-1 * diag(beta), one lane per column j (column j depends only on itself).
__device__ void chunk_solve(ChunkSmem& sm, int j) {
    float col[kC];
#pragma unroll
    for (int i = 0; i < kC; ++i) {
        float t = (i == j) ? 1.0f : 0.0f;
        if (i > j) {
            t = -sm.L[i][j];
            for (int m = j + 1; m < i; ++m)
                t -= sm.L[i][m] * col[m];
        }
        col[i] = t;
    }
#pragma unroll
    for (int i = 0; i < kC; ++i)
        sm.T[i][j] = col[i] * sm.beta[j];
}

// 4. Decay products (G still intact), warp 0 solves T.
__device__ void chunk_decay_products(const ChunkCtx& c, ChunkSmem& sm) {
    for (int idx = c.tid; idx < kC * kHD; idx += kChThreads) {
        const int i = idx / kHD;
        const int ch = idx % kHD;
        const float gi = sm.g[i][ch];
        const float kv = sm.k[i][ch];
        const float eg = __expf(gi);
        sm.e[i][ch] = eg * kv;
        sm.q[i][ch] *= eg;
        sm.k[i][ch] = kv * __expf(sm.g[kC - 1][ch] - gi);
    }
    sm.eC[c.tid] = __expf(sm.g[kC - 1][c.tid]);
    if (c.tid < kC)
        chunk_solve(sm, c.tid);
}

// 5. w = T (exp(G) . k) into g (G is dead past step 4).
__device__ void chunk_w(const ChunkCtx& c, ChunkSmem& sm) {
    for (int idx = c.tid; idx < kC * kHD; idx += kChThreads) {
        const int i = idx / kHD;
        const int ch = idx % kHD;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < kC; ++j)
            acc += sm.T[i][j] * sm.e[j][ch];
        sm.g[i][ch] = acc;
    }
}

// Sum over the 4 adjacent lanes sharing a value column.
__device__ __forceinline__ float column_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// 6a. v' = T v - w S for this thread's column d.
__device__ void chunk_vnew(const ChunkCtx& c, const ChunkSmem& sm, const float* S, int d, int s_base,
                           float* vn) {
    const KdaScanArgs& a = *c.a;
    float v[kC];
#pragma unroll
    for (int j = 0; j < kC; ++j) {
        v[j] = 0.0f;
        if (j < c.lc)
            v[j] = a.conv_f32[(chunk_row(c, j) * static_cast<size_t>(a.conv_channels)) +
                              (2 * static_cast<size_t>(c.HK)) + (static_cast<size_t>(c.h) * kHD) + d];
    }
#pragma unroll
    for (int i = 0; i < kC; ++i) {
        float u = 0.0f;
#pragma unroll
        for (int j = 0; j < kC; ++j)
            u += sm.T[i][j] * v[j];
        float ws = 0.0f;
#pragma unroll
        for (int s = 0; s < kChRows; ++s)
            ws += sm.g[i][s_base + s] * S[s];
        vn[i] = u - column_sum(ws);
    }
}

// 6b. o = (q . exp(G)) S + Aqk v' for this thread's column.
__device__ void chunk_out(const ChunkCtx& c, const ChunkSmem& sm, const float* S, const float* vn, int d,
                          int s_base, bool writer) {
#pragma unroll
    for (int i = 0; i < kC; ++i) {
        float qs = 0.0f;
#pragma unroll
        for (int s = 0; s < kChRows; ++s)
            qs += sm.q[i][s_base + s] * S[s];
        float o = column_sum(qs);
#pragma unroll
        for (int j = 0; j <= i; ++j)
            o += sm.Aqk[i][j] * vn[j];
        if (writer && i < c.lc)
            c.a->y[(chunk_row(c, i) * static_cast<size_t>(c.HK)) + (static_cast<size_t>(c.h) * kHD) + d] =
                __float2half(o);
    }
}

// 6c. S = exp(G_C) . S + (k . exp(G_C - G))^T v'.
__device__ void chunk_state(const ChunkSmem& sm, float* S, const float* vn, int s_base) {
#pragma unroll
    for (int s = 0; s < kChRows; ++s) {
        float acc = S[s] * sm.eC[s_base + s];
#pragma unroll
        for (int j = 0; j < kC; ++j)
            acc += sm.k[j][s_base + s] * vn[j];
        S[s] = acc;
    }
}

template <typename StateT>
__global__ void __launch_bounds__(kChThreads) kda_chunk_kernel(const __grid_constant__ KdaScanArgs a) {
    __shared__ ChunkSmem sm;
    ChunkCtx c{};
    c.a = &a;
    c.h = static_cast<int>(blockIdx.x);
    c.tid = static_cast<int>(threadIdx.x);
    c.sq = kda_resolve_seq(a, static_cast<int>(blockIdx.y));
    c.HK = a.n_heads * kHD;
    c.a_exp = -a.neg_a[c.h];
    c.scale = rsqrtf(static_cast<float>(kHD));
    const int part = c.tid % kChSplit;
    const int d = (static_cast<int>(blockIdx.z) * kVCols) + (c.tid / kChSplit);
    const int s_base = part * kChRows;
    StateT* state = kda_state_ptr<StateT>(a, static_cast<int>(blockIdx.y), c.h);

    float S[kChRows];
#pragma unroll
    for (int s = 0; s < kChRows; ++s)
        S[s] = kda_load(state + (static_cast<size_t>(s_base + s) * kHD) + d);

    for (c.c0 = 0; c.c0 < c.sq.n_rows; c.c0 += kC) {
        c.lc = min(kC, c.sq.n_rows - c.c0);
        chunk_load(c, sm);
        __syncthreads();
        chunk_norm_cumsum(c, sm);
        __syncthreads();
        chunk_pairs(c, sm);
        __syncthreads();
        chunk_decay_products(c, sm);
        __syncthreads();
        chunk_w(c, sm);
        __syncthreads();
        float vn[kC];
        chunk_vnew(c, sm, S, d, s_base, vn);
        chunk_out(c, sm, S, vn, d, s_base, part == 0);
        chunk_state(sm, S, vn, s_base);
        __syncthreads();  // every smem tile is rewritten by the next chunk
    }
#pragma unroll
    for (int s = 0; s < kChRows; ++s)
        kda_store(state + (static_cast<size_t>(s_base + s) * kHD) + d, S[s]);
}

}  // namespace

void kda_scan_chunk(const KdaScanArgs& a, cudaStream_t stream) {
    kda_check_args(a, "kda_scan_chunk");
    if (a.d_real_n)
        throw std::runtime_error("kda_scan_chunk: padded chunks (d_real_n) take kda_scan_recurrent");
    if (a.n_seq <= 0 || (!a.seq_row_offsets && a.n_tokens <= 0))
        return;
    const dim3 grid(static_cast<unsigned>(a.n_heads), static_cast<unsigned>(a.n_seq), kVSplit);
    if (a.state_bf16)
        kda_chunk_kernel<__nv_bfloat16><<<grid, kChThreads, 0, stream>>>(a);
    else
        kda_chunk_kernel<float><<<grid, kChThreads, 0, stream>>>(a);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
