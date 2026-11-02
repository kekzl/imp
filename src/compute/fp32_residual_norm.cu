// FP32-residual sandwich-norm chain fused per row (Gemma-3/4 decode): one CTA per row, 512 threads,
// thread t holds 8-element chunk t. Each step replays the reduction order of the kernel it replaces:
// rmsnorm_fp16_rowblock_kernel<1> (512 threads, one chunk each), rmsnorm_fp32_accum_to_fp16_kernel
// (256 threads, chunks t and t + 256), rmsnorm_fp32_in_kernel (512 threads, element-strided) and the
// rowblock NVFP4 producer quantize, so the output bits equal the unfused launches.
#include "compute/fp32_residual_norm.h"
#include "quant/nvfp4_pack.cuh"
#include "core/pdl_launch.cuh"
#include "core/pdl_device.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace imp {

namespace {

constexpr int kThreads = 512;
constexpr int kMaxD = 4096;  // accumulate kernel: 256 threads x 2 chunks x 8

struct Smem {
    float part[32];
    float bcast;
    float pair[256];
    float row[kMaxD];
};

__device__ __forceinline__ float4 ld4(const __half* p, int chunk) {
    return reinterpret_cast<const float4*>(p)[chunk];
}

// Warp xor-reduce of v, lane 0 of warps [0, n_warps) publish, warp 0 reduces lanes < n_warps.
// The result is valid in warp 0. block_reduce_sum and both row-block kernels use this order.
__device__ __forceinline__ float block_sum(float v, int n_warps, Smem& s) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    if (lane == 0 && warp < n_warps)
        s.part[warp] = v;
    __syncthreads();
    float t = 0.0f;
    if (warp == 0) {
        t = (lane < n_warps) ? s.part[lane] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            t += __shfl_xor_sync(0xFFFFFFFFu, t, o);
    }
    return t;
}

// rsqrt((total / d) + eps) of a block sum, broadcast to every thread.
__device__ __forceinline__ float block_inv_rms(float v, int n_warps, int d, float eps, Smem& s) {
    const float t = block_sum(v, n_warps, s);
    if (threadIdx.x == 0)
        s.bcast = rsqrtf((t / static_cast<float>(d)) + eps);
    __syncthreads();
    return s.bcast;
}

__device__ __forceinline__ float block_max(float v, Smem& s) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    if (lane == 0)
        s.part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < (kThreads >> 5)) ? s.part[lane] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
        if (lane == 0)
            s.bcast = m;
    }
    __syncthreads();
    return s.bcast;
}

// sanitize_fp16_kernel: +-Inf and NaN -> 0.
__device__ __forceinline__ float4 sanitize8(float4 v) {
    __half* h = reinterpret_cast<__half*>(&v);
#pragma unroll
    for (int k = 0; k < 8; ++k)
        if (__hisinf(h[k]) != 0 || __hisnan(h[k]))
            h[k] = __float2half(0.0f);
    return v;
}

// rmsnorm_fp16_rowblock_kernel<1>: sum of squares of the thread's chunk.
__device__ __forceinline__ float rowblock_ssq(const float4& v) {
    const half2* h = reinterpret_cast<const half2*>(&v);
    float sum_sq = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 f = __half22float2(h[k]);
        sum_sq += (f.x * f.x) + (f.y * f.y);
    }
    return sum_sq;
}

__device__ __forceinline__ float4 rowblock_apply(const float4& v, const float4& w, float inv_rms, float off) {
    const half2* xh = reinterpret_cast<const half2*>(&v);
    const half2* wh = reinterpret_cast<const half2*>(&w);
    float4 r;
    half2* rh = reinterpret_cast<half2*>(&r);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 xf = __half22float2(xh[k]);
        const float2 wf = __half22float2(wh[k]);
        rh[k] = __float22half2_rn(make_float2(xf.x * inv_rms * (wf.x + off), xf.y * inv_rms * (wf.y + off)));
    }
    return r;
}

// rmsnorm_fp32_accum_to_fp16_kernel: one chunk's term of the sum of squares.
__device__ __forceinline__ float accum_ssq(const float4& h4) {
    const half2* h2 = reinterpret_cast<const half2*>(&h4);
    const float2 f0 = __half22float2(h2[0]);
    const float2 f1 = __half22float2(h2[1]);
    const float2 f2 = __half22float2(h2[2]);
    const float2 f3 = __half22float2(h2[3]);
    return (f0.x * f0.x) + (f0.y * f0.y) + (f1.x * f1.x) + (f1.y * f1.y) + (f2.x * f2.x) + (f2.y * f2.y) +
           (f3.x * f3.x) + (f3.y * f3.y);
}

__device__ __forceinline__ void load8(float (&acc)[8], const float4* row, int t) {
    const float4 lo = row[static_cast<ptrdiff_t>(2) * t];
    const float4 hi = row[(static_cast<ptrdiff_t>(2) * t) + 1];
    acc[0] = lo.x;
    acc[1] = lo.y;
    acc[2] = lo.z;
    acc[3] = lo.w;
    acc[4] = hi.x;
    acc[5] = hi.y;
    acc[6] = hi.z;
    acc[7] = hi.w;
}

__device__ __forceinline__ void store8(float4* row, int t, const float (&acc)[8]) {
    row[static_cast<ptrdiff_t>(2) * t] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    row[(static_cast<ptrdiff_t>(2) * t) + 1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

// kTail input of the accumulate step: rmsnorm(sanitize(a), w_a) + rmsnorm(sanitize(b), w_b), FP16 add.
__device__ __forceinline__ float4 tail_input(const Fp32ResidArgs& a, int64_t off, bool act, const float4& wa,
                                             const float4& wb, Smem& s) {
    const int t = threadIdx.x;
    float4 va{}, vb{};
    if (act) {
        va = sanitize8(ld4(a.in_a + off, t));
        vb = sanitize8(ld4(a.in_b + off, t));
    }
    const float inv_a = block_inv_rms(act ? rowblock_ssq(va) : 0.0f, kThreads >> 5, a.d_model, a.eps, s);
    const float inv_b = block_inv_rms(act ? rowblock_ssq(vb) : 0.0f, kThreads >> 5, a.d_model, a.eps, s);
    float4 x{};
    if (act) {
        const float4 na = rowblock_apply(va, wa, inv_a, a.w_off);
        const float4 nb = rowblock_apply(vb, wb, inv_b, a.w_off);
        const half2* ha = reinterpret_cast<const half2*>(&na);
        const half2* hb = reinterpret_cast<const half2*>(&nb);
        half2* hx = reinterpret_cast<half2*>(&x);
#pragma unroll
        for (int k = 0; k < 4; ++k)
            hx[k] = __hadd2(ha[k], hb[k]);
    }
    return x;
}

// rmsnorm_fp32_accum_to_fp16_kernel (256 threads: thread u sums chunk u, then chunk u + 256):
// acc += rmsnorm(x) * w, returns the FP16 view acc * inv_scale (65000 row-max clamp).
__device__ __forceinline__ float4 accumulate(const float4& x, const float4& w4, float (&acc)[8], bool act,
                                             const Fp32ResidArgs& a, Smem& s) {
    const int t = threadIdx.x;
    const int d_vec = a.d_model >> 3;
    const float e = act ? accum_ssq(x) : 0.0f;
    if (t >= 256)
        s.pair[t - 256] = e;
    __syncthreads();
    float ssq = 0.0f;
    if (t < 256 && act)
        ssq += e;
    if (t < 256 && t + 256 < d_vec)
        ssq += s.pair[t];
    const float inv_rms = block_inv_rms(ssq, 8, a.d_model, a.eps, s);
    float local_max = 0.0f;
    if (act) {
        const half2* h2 = reinterpret_cast<const half2*>(&x);
        const half2* w2 = reinterpret_cast<const half2*>(&w4);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 f = __half22float2(h2[k]);
            const float2 w = __half22float2(w2[k]);
            const int j = 2 * k;
            acc[j] += f.x * inv_rms * (w.x + a.w_off);
            acc[j + 1] += f.y * inv_rms * (w.y + a.w_off);
        }
#pragma unroll
        for (int k = 0; k < 8; ++k)
            local_max = fmaxf(local_max, fabsf(acc[k]));
    }
    const float row_max = block_max(local_max, s);
    const float inv_scale = (row_max > 65000.0f) ? (65000.0f / row_max) : 1.0f;
    float4 h4;
    half2* oh = reinterpret_cast<half2*>(&h4);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int j = 2 * k;
        oh[k] = __floats2half2_rn(acc[j] * inv_scale, acc[j + 1] * inv_scale);
    }
    return h4;
}

// scale_fp16_by_fp16ptr_kernel + scale_fp32_by_fp16ptr_kernel on one chunk.
__device__ __forceinline__ void layer_scale(float4& h4, float (&acc)[8], __half sc) {
    const half2 s2 = __half2half2(sc);
    const float sf = __half2float(sc);
    half2* oh = reinterpret_cast<half2*>(&h4);
#pragma unroll
    for (int k = 0; k < 4; ++k)
        oh[k] = __hmul2(oh[k], s2);
#pragma unroll
    for (int k = 0; k < 8; ++k)
        acc[k] *= sf;
}

// rmsnorm_fp32_in_kernel<__half, __half> (thread u sums elements u, u + 512, ...) of acc into no;
// kQuant: the rowblock NVFP4 producer quantize of the rounded row (a thread pair per micro-block).
template <bool kQuant>
__device__ __forceinline__ void norm_out(const float (&acc)[8], const float4& wn, bool act, int64_t off,
                                         const Fp32ResidArgs& a, Smem& s) {
    const int t = threadIdx.x;
    const int d = a.d_model;
    if (act) {
#pragma unroll
        for (int k = 0; k < 8; ++k)
            s.row[(static_cast<ptrdiff_t>(8) * t) + k] = acc[k];
    }
    __syncthreads();
    float ssq = 0.0f;
    for (int i = t; i < d; i += kThreads) {
        const float v = s.row[i];
        ssq += v * v;
    }
    const float inv_rms = block_inv_rms(ssq, kThreads >> 5, d, a.eps, s);
    float vals[8];
    float amax = 0.0f;
    if (act) {
        float4 o4;
        __half* oh = reinterpret_cast<__half*>(&o4);
        const __half* wh = reinterpret_cast<const __half*>(&wn);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            oh[k] = __float2half(acc[k] * inv_rms * (__half2float(wh[k]) + a.w_off));
            vals[k] = __half2float(oh[k]);
            amax = fmaxf(amax, fabsf(vals[k]));
        }
        reinterpret_cast<float4*>(a.no_out + off)[t] = o4;
    }
    if constexpr (kQuant) {
        // d % 256 == 0 keeps every warp wholly active or wholly idle for the pair exchange.
        const float mb_amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
        if (!act)
            return;
        const int64_t packed_row = static_cast<int64_t>(blockIdx.x) * (d >> 1);
        const int64_t scales_row = static_cast<int64_t>(blockIdx.x) * (d >> 4);
        uint8_t s8;
        const float actual = nvfp4_encode_micro_scale(mb_amax, /*tensor_scale=*/1.0f, &s8);
        if ((t & 1) == 0)
            a.xq_scales[scales_row + (t >> 1)] = s8;
        const float inv = 1.0f / actual;
        uchar4 pb;
        pb.x = nvfp4_pack_pair_hw(vals[0] * inv, vals[1] * inv);
        pb.y = nvfp4_pack_pair_hw(vals[2] * inv, vals[3] * inv);
        pb.z = nvfp4_pack_pair_hw(vals[4] * inv, vals[5] * inv);
        pb.w = nvfp4_pack_pair_hw(vals[6] * inv, vals[7] * inv);
        *reinterpret_cast<uchar4*>(a.xq_packed + packed_row + (static_cast<int64_t>(t) * 4)) = pb;
    }
}

constexpr int kModeNorm = static_cast<int>(Fp32ResidMode::kNorm);
constexpr int kModePostAttn = static_cast<int>(Fp32ResidMode::kPostAttn);
constexpr int kModeTail = static_cast<int>(Fp32ResidMode::kTail);

template <int kMode, bool kQuant>
__global__ __launch_bounds__(kThreads) void fp32_residual_norm_kernel(Fp32ResidArgs a) {
    __shared__ Smem s;
    const int t = threadIdx.x;
    const bool act = t < (a.d_model >> 3);
    const int64_t off = static_cast<int64_t>(blockIdx.x) * a.d_model;
    const bool has_norm = a.w_norm != nullptr;

    // Norm weights and the layer scalar are immutable: load them before the grid dependency.
    float4 wa{}, wb{}, wacc{}, wn{};
    if (act && kMode == kModeTail) {
        wa = ld4(a.w_a, t);
        wb = ld4(a.w_b, t);
    }
    if (act && kMode != kModeNorm)
        wacc = ld4(a.w_accum, t);
    if (act && has_norm)
        wn = ld4(a.w_norm, t);
    const bool has_scale = kMode == kModeTail && a.layer_scale != nullptr;
    const __half lscale = has_scale ? *a.layer_scale : __float2half(1.0f);
    pdl_wait();
    pdl_trigger();  // scheduling only: the dependent GEMM prefetches its weights during this grid

    float acc[8] = {};
    float4* fp32_row = reinterpret_cast<float4*>(a.fp32 + off);
    if (act)
        load8(acc, fp32_row, t);
    if constexpr (kMode != kModeNorm) {
        float4 x{};
        if constexpr (kMode == kModePostAttn) {
            if (act)
                x = ld4(a.in_a + off, t);
        } else {
            x = tail_input(a, off, act, wa, wb, s);
        }
        float4 h4 = accumulate(x, wacc, acc, act, a, s);
        if (act && has_scale)
            layer_scale(h4, acc, lscale);
        if (act) {
            store8(fp32_row, t, acc);
            reinterpret_cast<float4*>(a.h_out + off)[t] = h4;
        }
        if (act && kMode == kModePostAttn && a.r_out != nullptr)
            reinterpret_cast<float4*>(a.r_out + off)[t] = h4;
    }
    if (has_norm)
        norm_out<kQuant>(acc, wn, act, off, a, s);
}

template <int kMode, bool kQuant>
void launch_mode(const Fp32ResidArgs& a, int rows, cudaStream_t stream) {
    pdl::enable_kernel(fp32_residual_norm_kernel<kMode, kQuant>);
    pdl::launch(fp32_residual_norm_kernel<kMode, kQuant>, dim3(rows), dim3(kThreads), 0, stream, a);
}

bool args_fit(Fp32ResidMode mode, const Fp32ResidArgs& a, bool quant) {
    const bool norm_ok = a.w_norm != nullptr && a.no_out != nullptr;
    switch (mode) {
        case Fp32ResidMode::kNorm:
            return norm_ok;
        case Fp32ResidMode::kPostAttn:
            return norm_ok && a.in_a != nullptr && a.w_accum != nullptr && a.h_out != nullptr;
        case Fp32ResidMode::kTail: {
            const bool inputs = a.in_a != nullptr && a.in_b != nullptr && a.w_a != nullptr &&
                                a.w_b != nullptr && a.w_accum != nullptr && a.h_out != nullptr;
            // The next norm is optional here; a quantize needs it.
            return inputs && (a.w_norm == nullptr || a.no_out != nullptr) && (!quant || a.w_norm != nullptr);
        }
    }
    return false;
}

}  // namespace

bool fp32_residual_norm_fits(Fp32ResidMode mode, int d_model) {
    // kTail's branch norms replay rmsnorm()'s row-block kernel<1>: 2048 <= d only.
    return d_model > 0 && (d_model & 255) == 0 && d_model <= kMaxD &&
           (mode != Fp32ResidMode::kTail || d_model >= 2048);
}

bool fp32_residual_norm(Fp32ResidMode mode, const Fp32ResidArgs& a, int rows, cudaStream_t stream) {
    const bool quant = a.xq_packed != nullptr && a.xq_scales != nullptr;
    if (rows <= 0 || a.fp32 == nullptr || !fp32_residual_norm_fits(mode, a.d_model) ||
        !args_fit(mode, a, quant))
        return false;
    if (mode == Fp32ResidMode::kNorm)
        quant ? launch_mode<kModeNorm, true>(a, rows, stream)
              : launch_mode<kModeNorm, false>(a, rows, stream);
    else if (mode == Fp32ResidMode::kPostAttn)
        quant ? launch_mode<kModePostAttn, true>(a, rows, stream)
              : launch_mode<kModePostAttn, false>(a, rows, stream);
    else
        quant ? launch_mode<kModeTail, true>(a, rows, stream)
              : launch_mode<kModeTail, false>(a, rows, stream);
    return true;
}

}  // namespace imp
