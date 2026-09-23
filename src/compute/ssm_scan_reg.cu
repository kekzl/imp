// Register-resident Mamba2 SSM scan: bit-identical to ssm_scan_kernel (ssm.cu), state kept in
// registers across the token loop instead of a global read-modify-write per token.
// Nemotron-3-Nano pp4096: ssm_scan_kernel 4.99 ms per 2048-token call, 78 % of the prefill kernel sum.
#include "compute/ssm_scan_reg.h"

#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace imp {
namespace {

constexpr int kThreads = 128;
constexpr int kPrefetch = 4;  // tokens of B/C/x/z in flight per thread
constexpr int kDtBlock = 32;  // tokens whose dt/a_bar one warp computes at once, one per lane

// Per-token operands of one thread: SPT consecutive B/C values of its s-chunk plus its x/z.
template <int SPT>
struct TokenIn {
    uint4 b[SPT / 8];
    uint4 c[SPT / 8];
    half x, z;
};

// Offsets are 32-bit: the launcher refuses n_tokens * row > INT_MAX.
template <int SPT, bool FUSE_GATE>
__device__ __forceinline__ void load_token(TokenIn<SPT>& in, const half* __restrict__ b_base,
                                           const half* __restrict__ c_base, const half* __restrict__ x_base,
                                           const half* __restrict__ z_base, int t, int bc_size,
                                           int inner_size) {
    const uint4* b = reinterpret_cast<const uint4*>(b_base + t * bc_size);
    const uint4* c = reinterpret_cast<const uint4*>(c_base + t * bc_size);
#pragma unroll
    for (int v = 0; v < SPT / 8; ++v) {
        in.b[v] = __ldg(b + v);
        in.c[v] = __ldg(c + v);
    }
    in.x = x_base[t * inner_size];
    if constexpr (FUSE_GATE)
        in.z = z_base[t * inner_size];
}

__device__ __forceinline__ float half_at(const uint4* v, int i) {
    return __half2float(reinterpret_cast<const half*>(v)[i]);
}

// Grid (n_heads, head_dim / (kThreads / S_TILES)); lanes s_tid = tid % S_TILES of one d share a
// warp segment, so the legacy smem tree (a[s] += a[s + stride]) becomes shfl_down, same order.
template <bool H_FP16, bool FUSE_GATE, int S_TILES, int SPT>
__global__ void ssm_scan_reg_kernel(const half* __restrict__ x, const half* __restrict__ B_in,
                                    const half* __restrict__ C_in, const half* __restrict__ dt_raw,
                                    const float* __restrict__ A_log, const float* __restrict__ D_skip,
                                    const float* __restrict__ dt_bias, void* __restrict__ h_state,
                                    half* __restrict__ y, const half* __restrict__ z, int n_tokens,
                                    int n_heads, int head_dim_ssm, int n_groups,
                                    const int* __restrict__ d_real_n, void* __restrict__ h_snap,
                                    const int* __restrict__ d_snap_n) {
    static_assert(SPT % 2 == 0, "state rounding runs in pairs");
    constexpr int kState = S_TILES * SPT;
    if (n_tokens <= 0)
        return;
    const int h = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int s_tid = threadIdx.x % S_TILES;
    const int d = blockIdx.y * (kThreads / S_TILES) + threadIdx.x / S_TILES;
    const int real_n = d_real_n ? min(n_tokens, __ldg(d_real_n)) : n_tokens;
    const int snap_n = (h_snap && d_snap_n) ? min(n_tokens, __ldg(d_snap_n)) : 0;

    const int g = h / (n_heads / n_groups);
    const float a_log_h = A_log[h];
    const float d_val = D_skip[h];
    const float dt_b = dt_bias[h];
    const int inner_size = n_heads * head_dim_ssm;
    const int bc_size = n_groups * kState;
    const int s_start = s_tid * SPT;
    const half* b_base = B_in + g * kState + s_start;
    const half* c_base = C_in + g * kState + s_start;
    const half* x_base = x + h * head_dim_ssm + d;
    const half* z_base = FUSE_GATE ? z + h * head_dim_ssm + d : nullptr;
    half* y_base = y + h * head_dim_ssm + d;
    const int64_t h_base = static_cast<int64_t>(h) * kState * head_dim_ssm + d;

    float hs[SPT];
#pragma unroll
    for (int i = 0; i < SPT; ++i) {
        const int64_t idx = h_base + static_cast<int64_t>(s_start + i) * head_dim_ssm;
        if constexpr (H_FP16)
            hs[i] = __half2float(static_cast<const half*>(h_state)[idx]);
        else
            hs[i] = static_cast<const float*>(h_state)[idx];
    }

    // dt_val / a_bar depend on (t, h) only: lane l computes token blk + l, step() reads it by
    // shuffle. Same expressions as ssm_scan_kernel (bit-identity is the contract).
    half dt_pref = dt_raw[min(lane, n_tokens - 1) * n_heads + h];
    float dtv_l = 0.0f, abar_l = 0.0f;
    auto refresh_dt = [&](int blk) {
        float dt_val = __half2float(dt_pref) + dt_b;
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        dtv_l = dt_val;
        abar_l = expf(dt_val * a_log_h);
        dt_pref = dt_raw[min(blk + kDtBlock + lane, n_tokens - 1) * n_heads + h];
    };

    auto step = [&](const TokenIn<SPT>& cur, int t, bool advance, bool check_snap) {
        const float dt_val = __shfl_sync(0xffffffffu, dtv_l, t & (kDtBlock - 1));
        const float a_bar = __shfl_sync(0xffffffffu, abar_l, t & (kDtBlock - 1));
        const float x_val = __half2float(cur.x);
        // Contraction pinned to the legacy SASS: h = fma(x*dt, b, a_bar*h), y = fma(c, h, y).
        const float dtx = __fmul_rn(x_val, dt_val);
        float hn[SPT];
        float y_partial = 0.0f;
#pragma unroll
        for (int i = 0; i < SPT; ++i) {
            hn[i] = __fmaf_rn(dtx, half_at(cur.b, i), __fmul_rn(a_bar, hs[i]));
            y_partial = __fmaf_rn(half_at(cur.c, i), hn[i], y_partial);
        }
        if (advance) {
#pragma unroll
            for (int i = 0; i < SPT; i += 2) {
                if constexpr (H_FP16) {
                    const float2 r = __half22float2(__floats2half2_rn(hn[i], hn[i + 1]));
                    hs[i] = r.x;
                    hs[i + 1] = r.y;
                } else {
                    hs[i] = hn[i];
                    hs[i + 1] = hn[i + 1];
                }
            }
        }
        if (check_snap && t == snap_n - 1) {
#pragma unroll
            for (int i = 0; i < SPT; ++i) {
                const int64_t idx = h_base + static_cast<int64_t>(s_start + i) * head_dim_ssm;
                if constexpr (H_FP16)
                    static_cast<half*>(h_snap)[idx] = __float2half(hn[i]);
                else
                    static_cast<float*>(h_snap)[idx] = hn[i];
            }
        }
#pragma unroll
        for (int off = S_TILES / 2; off > 0; off >>= 1)
            y_partial += __shfl_down_sync(0xffffffffu, y_partial, off, S_TILES);
        float y_val = y_partial + d_val * x_val;
        if constexpr (FUSE_GATE) {
            float z_val = __half2float(cur.z);
            z_val = z_val / (1.0f + expf(-z_val));
            y_val *= z_val;
        }
        if (s_tid == 0)
            y_base[t * inner_size] = __float2half(y_val);
    };

    // Ring slot j holds token t0 + j; it is consumed first and refilled with t0 + j + kPrefetch
    // (clamped to the last token, a harmless re-read) so no register copy is needed.
    TokenIn<SPT> ring[kPrefetch];
#pragma unroll
    for (int j = 0; j < kPrefetch; ++j)
        load_token<SPT, FUSE_GATE>(ring[j], b_base, c_base, x_base, z_base, min(j, n_tokens - 1), bc_size,
                                   inner_size);
    auto group = [&](int t0, bool hot) {
#pragma unroll
        for (int j = 0; j < kPrefetch; ++j) {
            if (hot)
                step(ring[j], t0 + j, true, false);
            else
                step(ring[j], t0 + j, t0 + j < real_n, true);
            load_token<SPT, FUSE_GATE>(ring[j], b_base, c_base, x_base, z_base,
                                       min(t0 + j + kPrefetch, n_tokens - 1), bc_size, inner_size);
        }
    };
    static_assert(kDtBlock % kPrefetch == 0, "a prefetch group never straddles a dt block");
    const int n_full = n_tokens - n_tokens % kPrefetch;
    for (int t0 = 0; t0 < n_full; t0 += kPrefetch) {
        if ((t0 & (kDtBlock - 1)) == 0)
            refresh_dt(t0);
        const bool snap_here = snap_n - 1 >= t0 && snap_n - 1 < t0 + kPrefetch;
        if (t0 + kPrefetch <= real_n && !snap_here)
            group(t0, true);  // the hot path
        else
            group(t0, false);
    }
    if (n_full < n_tokens) {
        if ((n_full & (kDtBlock - 1)) == 0)
            refresh_dt(n_full);
#pragma unroll
        for (int j = 0; j < kPrefetch; ++j)
            if (n_full + j < n_tokens)
                step(ring[j], n_full + j, n_full + j < real_n, true);
    }

    if (real_n > 0) {
#pragma unroll
        for (int i = 0; i < SPT; ++i) {
            const int64_t idx = h_base + static_cast<int64_t>(s_start + i) * head_dim_ssm;
            if constexpr (H_FP16)
                static_cast<half*>(h_state)[idx] = __float2half(hs[i]);
            else
                static_cast<float*>(h_state)[idx] = hs[i];
        }
    }
}

template <int S_TILES, int SPT>
void launch(const SsmScanArgs& a, bool fp16, bool fused) {
    const dim3 grid(a.n_heads, a.head_dim_ssm / (kThreads / S_TILES));
#define IMP_SSM_REG(H16, FG)                                                                                 \
    ssm_scan_reg_kernel<H16, FG, S_TILES, SPT>                                                               \
        <<<grid, kThreads, 0, a.stream>>>(a.x, a.B, a.C, a.dt, a.A_log, a.D, a.dt_bias, a.h_state, a.y, a.z, \
                                          a.n_tokens, a.n_heads, a.head_dim_ssm, a.n_groups, a.d_real_n,     \
                                          a.h_snap, a.d_snap_n)
    if (fp16 && fused)
        IMP_SSM_REG(true, true);
    else if (fp16)
        IMP_SSM_REG(true, false);
    else if (fused)
        IMP_SSM_REG(false, true);
    else
        IMP_SSM_REG(false, false);
#undef IMP_SSM_REG
    IMP_CUDA_CHECK_LAUNCH();
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

bool ssm_scan_reg_launch(const SsmScanArgs& a, int s_tiles, bool fp16) {
    // s_tiles is the legacy launcher's choice; bit-identity needs the same chunking and tree.
    const int spt = a.state_size / s_tiles;
    const int d_per_cta = kThreads / s_tiles;
    const int64_t row = std::max<int64_t>(static_cast<int64_t>(a.n_groups) * a.state_size,
                                          static_cast<int64_t>(a.n_heads) * a.head_dim_ssm);
    if (a.state_size % s_tiles != 0 || a.head_dim_ssm % d_per_cta != 0 || a.n_groups <= 0 ||
        a.n_heads % a.n_groups != 0 || !aligned16(a.B) || !aligned16(a.C) ||
        (a.n_groups * a.state_size) % 8 != 0 || static_cast<int64_t>(a.n_tokens) * row > INT32_MAX)
        return false;
    const bool fused = a.z != nullptr;
    if (s_tiles == 16 && spt == 8)
        launch<16, 8>(a, fp16, fused);  // Nemotron-H: head_dim 64, state 128
    else if (s_tiles == 8 && spt == 16)
        launch<8, 16>(a, fp16, fused);  // head_dim 128, state 128
    else
        return false;
    return true;
}

}  // namespace imp
