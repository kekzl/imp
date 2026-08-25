#include "compute/activation.h"
#include "quant/nvfp4_pack.cuh"
#include "runtime/pdl.h"
#include "core/tensor.h"
#include "core/logging.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace imp {

// --------------------------------------------------------------------------
// SwiGLU FP32 vectorized kernel (float4 path)
// out = silu(gate) * up = gate * sigmoid(gate) * up
// --------------------------------------------------------------------------
__global__ void swiglu_fp32_vec4_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                        float* __restrict__ out, int64_t n) {
    const int64_t vec_n = n / 4;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < vec_n) {
        float4 g = reinterpret_cast<const float4*>(gate)[idx];
        float4 u = reinterpret_cast<const float4*>(up)[idx];
        float4 o;
        o.x = g.x / (1.0f + __expf(-g.x)) * u.x;
        o.y = g.y / (1.0f + __expf(-g.y)) * u.y;
        o.z = g.z / (1.0f + __expf(-g.z)) * u.z;
        o.w = g.w / (1.0f + __expf(-g.w)) * u.w;
        reinterpret_cast<float4*>(out)[idx] = o;
    }

    // Handle tail elements
    const int64_t tail_start = vec_n * 4;
    // Let one thread handle the remaining elements
    if (idx == vec_n) {
        for (int64_t i = tail_start; i < n; ++i) {
            float g = gate[i];
            out[i] = g / (1.0f + __expf(-g)) * up[i];
        }
    }
}

// --------------------------------------------------------------------------
// SwiGLU FP32 scalar kernel
// --------------------------------------------------------------------------
__global__ void swiglu_fp32_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                   float* __restrict__ out, int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float sigmoid_g = 1.0f / (1.0f + __expf(-g));
        out[idx] = g * sigmoid_g * up[idx];
    }
}

// --------------------------------------------------------------------------
// SwiGLU FP16 kernel (load half, compute in float, store half)
// Processes 2 elements at a time using half2 / float conversion
// --------------------------------------------------------------------------
__global__ void swiglu_fp16_kernel(const __half* __restrict__ gate, const __half* __restrict__ up,
                                   __half* __restrict__ out, int64_t n) {
    const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        float2 gf = __half22float2(*reinterpret_cast<const __half2*>(gate + idx));
        float2 uf = __half22float2(*reinterpret_cast<const __half2*>(up + idx));

        float o0 = gf.x / (1.0f + __expf(-gf.x)) * uf.x;
        float o1 = gf.y / (1.0f + __expf(-gf.y)) * uf.y;

        *reinterpret_cast<__half2*>(out + idx) = __float22half2_rn(make_float2(o0, o1));
    } else if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        out[idx] = __float2half(g / (1.0f + __expf(-g)) * u);
    }
}

// --------------------------------------------------------------------------
// GeGLU kernels: out = gelu_tanh(gate) * up  (Gemma-3 activation)
// --------------------------------------------------------------------------

__global__ void geglu_fp16_kernel(const __half* __restrict__ gate, const __half* __restrict__ up,
                                  __half* __restrict__ out, int64_t n) {
    constexpr float SQRT_2_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;

    const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        float2 gf = __half22float2(*reinterpret_cast<const __half2*>(gate + idx));
        float2 uf = __half22float2(*reinterpret_cast<const __half2*>(up + idx));

        float gelu0 = gf.x * 0.5f * (1.0f + tanhf(SQRT_2_PI * (gf.x + COEFF * gf.x * gf.x * gf.x)));
        float gelu1 = gf.y * 0.5f * (1.0f + tanhf(SQRT_2_PI * (gf.y + COEFF * gf.y * gf.y * gf.y)));

        // Clamp to FP16 range to avoid Inf (products can exceed 65504 during prefill)
        float r0 = fminf(fmaxf(gelu0 * uf.x, -65504.0f), 65504.0f);
        float r1 = fminf(fmaxf(gelu1 * uf.y, -65504.0f), 65504.0f);
        *reinterpret_cast<__half2*>(out + idx) = __float22half2_rn(make_float2(r0, r1));
    } else if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float gelu_g = g * 0.5f * (1.0f + tanhf(SQRT_2_PI * (g + COEFF * g * g * g)));
        float result = fminf(fmaxf(gelu_g * u, -65504.0f), 65504.0f);
        out[idx] = __float2half(result);
    }
}

__global__ void geglu_fp32_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                  float* __restrict__ out, int64_t n) {
    constexpr float SQRT_2_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float gelu_g = g * 0.5f * (1.0f + tanhf(SQRT_2_PI * (g + COEFF * g * g * g)));
        out[idx] = gelu_g * up[idx];
    }
}

__global__ void geglu_fp32_vec4_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                       float* __restrict__ out, int64_t n) {
    constexpr float SQRT_2_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;

    const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 g4 = reinterpret_cast<const float4*>(gate)[idx / 4];
        float4 u4 = reinterpret_cast<const float4*>(up)[idx / 4];
        float4 o4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float g = (&g4.x)[i];
            float gelu_g = g * 0.5f * (1.0f + tanhf(SQRT_2_PI * (g + COEFF * g * g * g)));
            (&o4.x)[i] = gelu_g * (&u4.x)[i];
        }
        reinterpret_cast<float4*>(out)[idx / 4] = o4;
    }
}

// --------------------------------------------------------------------------
// gpt-oss clamped GLU kernels (issue #547):
//   gate_c = min(gate, 7);  up_c = clamp(up, -7, 7)
//   out = (up_c + 1) * gate_c * sigmoid(1.702 * gate_c)
// --------------------------------------------------------------------------
__device__ __forceinline__ float gpt_oss_glu_elem(float g, float u) {
    constexpr float kLimit = 7.0f;
    constexpr float kAlpha = 1.702f;
    g = fminf(g, kLimit);
    u = fminf(fmaxf(u, -kLimit), kLimit);
    float glu = g / (1.0f + __expf(-kAlpha * g));
    return (u + 1.0f) * glu;
}

__global__ void gpt_oss_glu_fp16_kernel(const __half* __restrict__ gate, const __half* __restrict__ up,
                                        __half* __restrict__ out, int64_t n) {
    const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        float2 gf = __half22float2(*reinterpret_cast<const __half2*>(gate + idx));
        float2 uf = __half22float2(*reinterpret_cast<const __half2*>(up + idx));
        *reinterpret_cast<__half2*>(out + idx) = __float22half2_rn(
            make_float2(gpt_oss_glu_elem(gf.x, uf.x), gpt_oss_glu_elem(gf.y, uf.y)));
    } else if (idx < n) {
        out[idx] = __float2half(gpt_oss_glu_elem(__half2float(gate[idx]), __half2float(up[idx])));
    }
}

__global__ void gpt_oss_glu_fp32_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                        float* __restrict__ out, int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = gpt_oss_glu_elem(gate[idx], up[idx]);
}

__global__ void gpt_oss_glu_fp32_vec4_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                             float* __restrict__ out, int64_t n) {
    const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 g4 = reinterpret_cast<const float4*>(gate)[idx / 4];
        float4 u4 = reinterpret_cast<const float4*>(up)[idx / 4];
        float4 o4;
#pragma unroll
        for (int i = 0; i < 4; i++)
            (&o4.x)[i] = gpt_oss_glu_elem((&g4.x)[i], (&u4.x)[i]);
        reinterpret_cast<float4*>(out)[idx / 4] = o4;
    }
}

// --------------------------------------------------------------------------
// GELU FP32 vectorized kernel
// gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// --------------------------------------------------------------------------
static constexpr float GELU_SQRT_2_OVER_PI = 0.7978845608028654f;
static constexpr float GELU_COEFF = 0.044715f;

__global__ void gelu_fp32_vec4_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
    const int64_t vec_n = n / 4;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < vec_n) {
        float4 v = reinterpret_cast<const float4*>(x)[idx];
        float4 o;

#define GELU_ELEM(val) \
    (val) * 0.5f * (1.0f + tanhf(GELU_SQRT_2_OVER_PI * ((val) + GELU_COEFF * (val) * (val) * (val))))

        o.x = GELU_ELEM(v.x);
        o.y = GELU_ELEM(v.y);
        o.z = GELU_ELEM(v.z);
        o.w = GELU_ELEM(v.w);

#undef GELU_ELEM

        reinterpret_cast<float4*>(out)[idx] = o;
    }

    // Tail
    if (idx == vec_n) {
        for (int64_t i = vec_n * 4; i < n; ++i) {
            float v = x[i];
            out[i] = v * 0.5f * (1.0f + tanhf(GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v)));
        }
    }
}

// --------------------------------------------------------------------------
// GELU FP32 scalar kernel
// --------------------------------------------------------------------------
__global__ void gelu_fp32_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = v * 0.5f * (1.0f + tanhf(GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v)));
    }
}

// --------------------------------------------------------------------------
// GELU FP16 kernel (load half2, compute in float, store half2)
// --------------------------------------------------------------------------
__global__ void gelu_fp16_kernel(const __half* __restrict__ x, __half* __restrict__ out, int64_t n) {
    const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        float2 vf = __half22float2(*reinterpret_cast<const __half2*>(x + idx));

        float o0 = vf.x * 0.5f *
                   (1.0f + tanhf(GELU_SQRT_2_OVER_PI * (vf.x + GELU_COEFF * vf.x * vf.x * vf.x)));
        float o1 = vf.y * 0.5f *
                   (1.0f + tanhf(GELU_SQRT_2_OVER_PI * (vf.y + GELU_COEFF * vf.y * vf.y * vf.y)));

        *reinterpret_cast<__half2*>(out + idx) = __float22half2_rn(make_float2(o0, o1));
    } else if (idx < n) {
        float v = __half2float(x[idx]);
        out[idx] = __float2half(v * 0.5f *
                                (1.0f + tanhf(GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v))));
    }
}

// --------------------------------------------------------------------------
// Host dispatch helpers — gated (2-input) and unary (1-input) activation
// launch with FP32 vec4/scalar + FP16 half2 paths.
// --------------------------------------------------------------------------

// Gated activation dispatch: out = act(gate) * up
// Handles FP32 (vec4 + scalar fallback) and FP16 (half2, PDL-enabled).
static void dispatch_gated_activation(const Tensor& gate, const Tensor& up, Tensor& out, int64_t n,
                                      void (*fp32_vec4)(const float*, const float*, float*, int64_t),
                                      void (*fp32_scalar)(const float*, const float*, float*, int64_t),
                                      void (*fp16_kernel)(const __half*, const __half*, __half*, int64_t),
                                      bool pdl_enabled, cudaStream_t stream) {
    const int block = 256;
    switch (gate.qtype) {
        case QType::F32:
            if (n % 4 == 0 && n >= 4) {
                const int grid = static_cast<int>((n / 4 + block - 1) / block);
                fp32_vec4<<<grid, block, 0, stream>>>(static_cast<const float*>(gate.data),
                                                      static_cast<const float*>(up.data),
                                                      static_cast<float*>(out.data), n);
                IMP_CUDA_CHECK_LAUNCH();
            } else {
                const int grid = static_cast<int>((n + block - 1) / block);
                fp32_scalar<<<grid, block, 0, stream>>>(static_cast<const float*>(gate.data),
                                                        static_cast<const float*>(up.data),
                                                        static_cast<float*>(out.data), n);
                IMP_CUDA_CHECK_LAUNCH();
            }
            break;
        case QType::F16: {
            const int64_t half_n = (n + 1) / 2;
            const int grid = static_cast<int>((half_n + block - 1) / block);
            if (pdl_enabled) {
                pdl::launch(fp16_kernel, dim3(grid), dim3(block), size_t(0), stream,
                            static_cast<const __half*>(gate.data), static_cast<const __half*>(up.data),
                            static_cast<__half*>(out.data), n);
            } else {
                fp16_kernel<<<grid, block, 0, stream>>>(static_cast<const __half*>(gate.data),
                                                        static_cast<const __half*>(up.data),
                                                        static_cast<__half*>(out.data), n);
                IMP_CUDA_CHECK_LAUNCH();
            }
            break;
        }
        default:
            break;
    }
}

// --------------------------------------------------------------------------
// Fused SwiGLU + NVFP4 activation quantize — the batched-decode producer
// fusion for the down-projection input. One thread per 16-value micro-block:
// silu(gate)*up in float (same arithmetic as swiglu_fp16_kernel), rounded to
// FP16 and stored (bit-identical `out`), then the ROUNDED values are packed
// to NVFP4 nibbles + FP8 micro-scale (plain layout, tensor_scale 1.0 —
// bit-identical to quantize_fp16_to_nvfp4_into on the stored FP16). Kills
// the down GEMM's quantize launch + its [M,K] FP16 re-read.
// --------------------------------------------------------------------------
__global__ void swiglu_fp16_nvfp4_kernel(const __half* __restrict__ gate, const __half* __restrict__ up,
                                         __half* __restrict__ out, uint8_t* __restrict__ xq_packed,
                                         uint8_t* __restrict__ xq_scales, int64_t total_mb) {
    const int64_t mb = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (mb >= total_mb)
        return;
    const int64_t base = mb * 16;

    float4 gv[2], uv[2];
    gv[0] = reinterpret_cast<const float4*>(gate + base)[0];
    gv[1] = reinterpret_cast<const float4*>(gate + base)[1];
    uv[0] = reinterpret_cast<const float4*>(up + base)[0];
    uv[1] = reinterpret_cast<const float4*>(up + base)[1];

    float vals[16];
    float amax = 0.0f;
    float4 ov[2];
#pragma unroll
    for (int v = 0; v < 2; ++v) {
        const half2* g2 = reinterpret_cast<const half2*>(&gv[v]);
        const half2* u2 = reinterpret_cast<const half2*>(&uv[v]);
        half2* o2 = reinterpret_cast<half2*>(&ov[v]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 gf = __half22float2(g2[k]);
            const float2 uf = __half22float2(u2[k]);
            const float o0 = gf.x / (1.0f + __expf(-gf.x)) * uf.x;
            const float o1 = gf.y / (1.0f + __expf(-gf.y)) * uf.y;
            o2[k] = __float22half2_rn(make_float2(o0, o1));
            const float2 rf = __half22float2(o2[k]);
            const int idx = v * 8 + k * 2;
            vals[idx] = rf.x;
            vals[idx + 1] = rf.y;
            amax = fmaxf(amax, fmaxf(fabsf(rf.x), fabsf(rf.y)));
        }
    }
    reinterpret_cast<float4*>(out + base)[0] = ov[0];
    reinterpret_cast<float4*>(out + base)[1] = ov[1];

    uint8_t s8;
    const float actual = nvfp4_encode_micro_scale(amax, /*tensor_scale=*/1.0f, &s8);
    xq_scales[mb] = s8;
    const float inv = 1.0f / actual;
    uint2 pk;
    uint8_t* pb = reinterpret_cast<uint8_t*>(&pk);
#pragma unroll
    for (int k = 0; k < 8; ++k)
        pb[k] = nvfp4_pack_pair_hw(vals[k * 2] * inv, vals[k * 2 + 1] * inv);
    *reinterpret_cast<uint2*>(xq_packed + mb * 8) = pk;
}

// --------------------------------------------------------------------------
// Host dispatch: swiglu
// --------------------------------------------------------------------------
void swiglu(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const int64_t n = gate.numel();
    if (n == 0)
        return;
    dispatch_gated_activation(gate, up, out, n, swiglu_fp32_vec4_kernel, swiglu_fp32_kernel,
                              swiglu_fp16_kernel, true, stream);
}

// --------------------------------------------------------------------------
// Host dispatch: fused swiglu + NVFP4 quantize (see kernel comment).
// Returns false outside the fused envelope; caller falls back to swiglu()
// + the GEMM dispatch's own quantize.
// --------------------------------------------------------------------------
bool swiglu_quantize_nvfp4(const Tensor& gate, const Tensor& up, Tensor& out, uint8_t* xq_packed,
                           uint8_t* xq_scales, cudaStream_t stream) {
    if (gate.qtype != QType::F16 || up.qtype != QType::F16 || out.qtype != QType::F16)
        return false;
    const int64_t n = gate.numel();
    if (n == 0 || (n & 15) != 0)
        return false;
    const int64_t total_mb = n >> 4;
    const int block = 256;
    const int grid = static_cast<int>((total_mb + block - 1) / block);
    pdl::launch(swiglu_fp16_nvfp4_kernel, dim3(grid), dim3(block), size_t(0), stream,
                static_cast<const __half*>(gate.data), static_cast<const __half*>(up.data),
                static_cast<__half*>(out.data), xq_packed, xq_scales, total_mb);
    return true;
}

// --------------------------------------------------------------------------
// Host dispatch: geglu
// --------------------------------------------------------------------------
void geglu(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const int64_t n = gate.numel();
    if (n == 0)
        return;
    dispatch_gated_activation(gate, up, out, n, geglu_fp32_vec4_kernel, geglu_fp32_kernel, geglu_fp16_kernel,
                              true, stream);
}

// --------------------------------------------------------------------------
// Host dispatch: gpt_oss_glu
// --------------------------------------------------------------------------
void gpt_oss_glu(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const int64_t n = gate.numel();
    if (n == 0)
        return;
    dispatch_gated_activation(gate, up, out, n, gpt_oss_glu_fp32_vec4_kernel, gpt_oss_glu_fp32_kernel,
                              gpt_oss_glu_fp16_kernel, true, stream);
}

// --------------------------------------------------------------------------
// Host dispatch: gelu
// --------------------------------------------------------------------------
void gelu(const Tensor& x, Tensor& out, cudaStream_t stream) {
    const int64_t n = x.numel();
    if (n == 0)
        return;

    const int block = 256;
    switch (x.qtype) {
        case QType::F32:
            if (n % 4 == 0 && n >= 4) {
                const int grid = static_cast<int>((n / 4 + block - 1) / block);
                gelu_fp32_vec4_kernel<<<grid, block, 0, stream>>>(static_cast<const float*>(x.data),
                                                                  static_cast<float*>(out.data), n);
                IMP_CUDA_CHECK_LAUNCH();
            } else {
                const int grid = static_cast<int>((n + block - 1) / block);
                gelu_fp32_kernel<<<grid, block, 0, stream>>>(static_cast<const float*>(x.data),
                                                             static_cast<float*>(out.data), n);
                IMP_CUDA_CHECK_LAUNCH();
            }
            break;
        case QType::F16: {
            const int64_t half_n = (n + 1) / 2;
            const int grid = static_cast<int>((half_n + block - 1) / block);
            gelu_fp16_kernel<<<grid, block, 0, stream>>>(static_cast<const __half*>(x.data),
                                                         static_cast<__half*>(out.data), n);
            IMP_CUDA_CHECK_LAUNCH();
            break;
        }
        default:
            break;
    }
}

// --------------------------------------------------------------------------
// Shared-expert sigmoid gate (Qwen3-Next / Qwen3.6):
//   per row r: gate = sigmoid( sum_d x[r,d] * W[d] )
//   y[r,:]   *= gate
// One block per row, block reduces the dot product in shared memory and then
// rescales the entire row of y in place.
// --------------------------------------------------------------------------
__global__ void shared_expert_gate_scale_kernel(
    const __half* __restrict__ x,  // [n, d_model]
    const __half* __restrict__ W,  // [d_model] — FP16 (upload converts F32→FP16)
    __half* __restrict__ y,        // [n, d]  — scaled in place
    int d_model, int d) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_threads = blockDim.x;

    // Dot product x[row, :] * W[:] accumulated in FP32.
    const __half* xr = x + static_cast<int64_t>(row) * d_model;
    float local = 0.0f;
    for (int k = tid; k < d_model; k += block_threads) {
        local += __half2float(xr[k]) * __half2float(W[k]);
    }

    __shared__ float s_reduce[32];
    // Warp reduce
    unsigned mask = 0xffffffff;
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_xor_sync(mask, local, off);
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0)
        s_reduce[warp] = local;
    __syncthreads();

    // First warp reduces across warps
    int n_warps = block_threads >> 5;
    if (warp == 0) {
        local = (tid < n_warps) ? s_reduce[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            local += __shfl_xor_sync(mask, local, off);
        }
        if (tid == 0)
            s_reduce[0] = local;
    }
    __syncthreads();

    // Sigmoid
    float sum = s_reduce[0];
    float gate = 1.0f / (1.0f + __expf(-sum));
    __half gate_h = __float2half(gate);

    // Scale y[row, :] by gate in place
    __half* yr = y + static_cast<int64_t>(row) * d;
    for (int j = tid; j < d; j += block_threads) {
        yr[j] = __hmul(yr[j], gate_h);
    }
}

void shared_expert_gate_scale(const void* x_fp16, const void* W_fp16, void* y_fp16_inout, int n, int d_model,
                              int d, cudaStream_t stream) {
    if (n == 0)
        return;
    const int block = 256;
    shared_expert_gate_scale_kernel<<<n, block, 0, stream>>>(static_cast<const __half*>(x_fp16),
                                                             static_cast<const __half*>(W_fp16),
                                                             static_cast<__half*>(y_fp16_inout), d_model, d);
    IMP_CUDA_CHECK_LAUNCH();
}

// --------------------------------------------------------------------------
// PDL registration
// --------------------------------------------------------------------------
void activation_pdl_register() {
    pdl::enable(reinterpret_cast<const void*>(&swiglu_fp16_kernel));
    pdl::enable(reinterpret_cast<const void*>(&geglu_fp16_kernel));
}

}  // namespace imp
