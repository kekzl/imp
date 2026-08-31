// layernorm_rowblock.cu — the batched-decode (2..64 rows) FP16 RMSNorm
// family: one register-resident CTA per row, plus the producer-fusion
// variant that also emits the small-M NVFP4 activation quantize. Split out
// of layernorm.cu (one logical unit per TU; the hard-review size gate was
// the trigger, the recompile blast radius is the reason it stays split).
// Both kernels moved VERBATIM from layernorm.cu.
#include "compute/layernorm.h"
#include "quant/nvfp4_pack.cuh"
#include "runtime/pdl.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "compute/pdl_device.cuh"

namespace imp {

// --------------------------------------------------------------------------
// Row-block FP16 RMSNorm — the batched-decode variant (2 <= rows <= 64).
//
// At mbs<=32 the warp-per-row kernel launches 4 CTAs on 170 SMs and
// walks every row TWICE through DRAM (sum-of-squares pass + normalize
// pass): measured 6.3 us median for 0.65 MB of traffic at rows=32 d=5120,
// ~6% of DRAM bandwidth, pure latency (nsys 2026-08-25, the norms row of
// the concurrency-gap attribution). One CTA per row keeps the row
// REGISTER-resident across the reduction: read once, one block reduce,
// write once — and rows x 1 CTA puts every row's loads in flight at once.
// --------------------------------------------------------------------------
template <int kVecs>
__global__ void rmsnorm_fp16_rowblock_kernel(const __half* __restrict__ x, const __half* __restrict__ weight,
                                             __half* __restrict__ out, int d_model, float eps,
                                             float weight_offset) {
    const int d_vec = d_model >> 3;
    const float4* x_vec = reinterpret_cast<const float4*>(x + static_cast<int64_t>(blockIdx.x) * d_model);
    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* out_vec = reinterpret_cast<float4*>(out + static_cast<int64_t>(blockIdx.x) * d_model);

    float4 v[kVecs];
    float sum_sq = 0.0f;
    pdl_wait();  // first global read follows
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
        const int i = threadIdx.x + j * static_cast<int>(blockDim.x);
        if (i < d_vec) {
            v[j] = x_vec[i];
            const half2* h = reinterpret_cast<const half2*>(&v[j]);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 f = __half22float2(h[k]);
                sum_sq += f.x * f.x + f.y * f.y;
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFFu, sum_sq, off);
    __shared__ float s_part[32];
    __shared__ float s_inv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0)
        s_part[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < (static_cast<int>(blockDim.x) >> 5)) ? s_part[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_xor_sync(0xFFFFFFFFu, t, off);
        if (lane == 0)
            s_inv = rsqrtf(t / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv;
    pdl_trigger();  // inputs are in registers; only the weight read + stores remain
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
        const int i = threadIdx.x + j * static_cast<int>(blockDim.x);
        if (i < d_vec) {
            const float4 wv = w_vec[i];
            const half2* xh = reinterpret_cast<const half2*>(&v[j]);
            const half2* wh = reinterpret_cast<const half2*>(&wv);
            float4 result;
            half2* rh = reinterpret_cast<half2*>(&result);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 xf = __half22float2(xh[k]);
                const float2 wf = __half22float2(wh[k]);
                rh[k] = __float22half2_rn(make_float2(xf.x * inv_rms * (wf.x + weight_offset),
                                                      xf.y * inv_rms * (wf.y + weight_offset)));
            }
            out_vec[i] = result;
        }
    }
}

// --------------------------------------------------------------------------
// Row-block FP16 RMSNorm + NVFP4 activation quantize — the batched-decode
// producer fusion. Identical norm arithmetic to the kernel above; on top,
// each adjacent thread PAIR owns one 16-value micro-block (two consecutive
// float4s in every j-slice) and emits the packed nibbles + FP8 micro-scale
// the small-M NVFP4 GEMM reads, from the ROUNDED fp16 values — the separate
// quantize kernel reads the stored FP16 row, so bit-identity requires
// quantizing post-rounding. Kills one quantize launch + one [M,K] FP16
// re-read per consumer group (q/kv, gate/up, GDN in/z).
// Caller guarantees d_model % 256 == 0 (whole-warp pair activity per slice).
// --------------------------------------------------------------------------
template <int kVecs>
__global__ void rmsnorm_fp16_rowblock_nvfp4_kernel(const __half* __restrict__ x,
                                                   const __half* __restrict__ weight,
                                                   __half* __restrict__ out,
                                                   uint8_t* __restrict__ xq_packed,
                                                   uint8_t* __restrict__ xq_scales, int d_model,
                                                   float eps, float weight_offset) {
    const int d_vec = d_model >> 3;
    const float4* x_vec = reinterpret_cast<const float4*>(x + static_cast<int64_t>(blockIdx.x) * d_model);
    const float4* w_vec = reinterpret_cast<const float4*>(weight);
    float4* out_vec = reinterpret_cast<float4*>(out + static_cast<int64_t>(blockIdx.x) * d_model);

    float4 v[kVecs];
    float sum_sq = 0.0f;
    pdl_wait();  // first global read follows
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
        const int i = threadIdx.x + j * static_cast<int>(blockDim.x);
        if (i < d_vec) {
            v[j] = x_vec[i];
            const half2* h = reinterpret_cast<const half2*>(&v[j]);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 f = __half22float2(h[k]);
                sum_sq += f.x * f.x + f.y * f.y;
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_xor_sync(0xFFFFFFFFu, sum_sq, off);
    __shared__ float s_part[32];
    __shared__ float s_inv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0)
        s_part[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < (static_cast<int>(blockDim.x) >> 5)) ? s_part[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_xor_sync(0xFFFFFFFFu, t, off);
        if (lane == 0)
            s_inv = rsqrtf(t / static_cast<float>(d_model) + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv;
    pdl_trigger();  // inputs are in registers; only the weight read + stores remain

    const int64_t packed_row = static_cast<int64_t>(blockIdx.x) * (d_model >> 1);
    const int64_t scales_row = static_cast<int64_t>(blockIdx.x) * (d_model >> 4);
#pragma unroll
    for (int j = 0; j < kVecs; ++j) {
        const int i = threadIdx.x + j * static_cast<int>(blockDim.x);
        const bool active = i < d_vec;
        float vals[8];
        float amax = 0.0f;
        if (active) {
            const float4 wv = w_vec[i];
            const half2* xh = reinterpret_cast<const half2*>(&v[j]);
            const half2* wh = reinterpret_cast<const half2*>(&wv);
            float4 result;
            half2* rh = reinterpret_cast<half2*>(&result);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 xf = __half22float2(xh[k]);
                const float2 wf = __half22float2(wh[k]);
                rh[k] = __float22half2_rn(make_float2(xf.x * inv_rms * (wf.x + weight_offset),
                                                      xf.y * inv_rms * (wf.y + weight_offset)));
                const float2 rf = __half22float2(rh[k]);
                vals[k * 2] = rf.x;
                vals[k * 2 + 1] = rf.y;
                amax = fmaxf(amax, fmaxf(fabsf(rf.x), fabsf(rf.y)));
            }
            out_vec[i] = result;
        }
        // Pair-exchange OUTSIDE the activity guard; slices are whole-warp
        // active (d_model % 256 == 0), so a pair is never split.
        const float mb_amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
        if (active) {
            uint8_t s8;
            const float actual = nvfp4_encode_micro_scale(mb_amax, /*tensor_scale=*/1.0f, &s8);
            if ((i & 1) == 0)
                xq_scales[scales_row + (i >> 1)] = s8;
            const float inv = 1.0f / actual;
            uchar4 pb;
            pb.x = nvfp4_pack_pair_hw(vals[0] * inv, vals[1] * inv);
            pb.y = nvfp4_pack_pair_hw(vals[2] * inv, vals[3] * inv);
            pb.z = nvfp4_pack_pair_hw(vals[4] * inv, vals[5] * inv);
            pb.w = nvfp4_pack_pair_hw(vals[6] * inv, vals[7] * inv);
            *reinterpret_cast<uchar4*>(xq_packed + packed_row + static_cast<int64_t>(i) * 4) = pb;
        }
    }
}

// --------------------------------------------------------------------------
// Launcher for the plain row-block kernel — called from rmsnorm()'s
// batched-decode branch (layernorm.cu). Caller checked the envelope
// (F16, rows 2..64, d % 8 == 0, d_vec <= 1024).
// --------------------------------------------------------------------------
void rmsnorm_fp16_rowblock(const Tensor& x, const Tensor& weight, Tensor& out, int rows, int d_model,
                           float eps, cudaStream_t stream, float weight_offset) {
    if ((d_model >> 3) <= 512) {
        pdl::enable_kernel(rmsnorm_fp16_rowblock_kernel<1>);
        pdl::launch(rmsnorm_fp16_rowblock_kernel<1>, dim3(rows), dim3(512), 0, stream,
                    static_cast<const __half*>(x.data), static_cast<const __half*>(weight.data),
                    static_cast<__half*>(out.data), d_model, eps, weight_offset);
    } else {
        pdl::enable_kernel(rmsnorm_fp16_rowblock_kernel<2>);
        pdl::launch(rmsnorm_fp16_rowblock_kernel<2>, dim3(rows), dim3(512), 0, stream,
                    static_cast<const __half*>(x.data), static_cast<const __half*>(weight.data),
                    static_cast<__half*>(out.data), d_model, eps, weight_offset);
    }
}

// --------------------------------------------------------------------------
// Host dispatch: rmsnorm + NVFP4 activation quantize (producer fusion).
// Returns false when the shape is outside the fused kernel's envelope —
// the caller then runs the plain rmsnorm() and lets the GEMM dispatch
// quantize as before.
// --------------------------------------------------------------------------
bool rmsnorm_nvfp4(const Tensor& x, const Tensor& weight, Tensor& out, uint8_t* xq_packed,
                   uint8_t* xq_scales, float eps, cudaStream_t stream, float weight_offset) {
    const int rows = static_cast<int>(x.shape[0]);
    const int d_model = static_cast<int>(x.shape[1]);
    if (x.qtype != QType::F16 || rows < 2 || rows > 64 || (d_model & 255) != 0 ||
        (d_model >> 3) > 1024)
        return false;
    if ((d_model >> 3) <= 512) {
        pdl::enable_kernel(rmsnorm_fp16_rowblock_nvfp4_kernel<1>);
        pdl::launch(rmsnorm_fp16_rowblock_nvfp4_kernel<1>, dim3(rows), dim3(512), 0, stream,
                    static_cast<const __half*>(x.data), static_cast<const __half*>(weight.data),
                    static_cast<__half*>(out.data), xq_packed, xq_scales, d_model, eps, weight_offset);
    } else {
        pdl::enable_kernel(rmsnorm_fp16_rowblock_nvfp4_kernel<2>);
        pdl::launch(rmsnorm_fp16_rowblock_nvfp4_kernel<2>, dim3(rows), dim3(512), 0, stream,
                    static_cast<const __half*>(x.data), static_cast<const __half*>(weight.data),
                    static_cast<__half*>(out.data), xq_packed, xq_scales, d_model, eps, weight_offset);
    }
    return true;
}

}  // namespace imp
