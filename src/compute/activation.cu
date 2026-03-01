#include "compute/activation.h"
#include "runtime/pdl.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace imp {

// --------------------------------------------------------------------------
// SwiGLU FP32 vectorized kernel (float4 path)
// out = silu(gate) * up = gate * sigmoid(gate) * up
// --------------------------------------------------------------------------
__global__ void swiglu_fp32_vec4_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int64_t n)
{
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
__global__ void swiglu_fp32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int64_t n)
{
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
__global__ void swiglu_fp16_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    int64_t n)
{
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

__global__ void geglu_fp16_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    int64_t n)
{
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

__global__ void geglu_fp32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int64_t n)
{
    constexpr float SQRT_2_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float gelu_g = g * 0.5f * (1.0f + tanhf(SQRT_2_PI * (g + COEFF * g * g * g)));
        out[idx] = gelu_g * up[idx];
    }
}

// --------------------------------------------------------------------------
// GELU FP32 vectorized kernel
// gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// --------------------------------------------------------------------------
static constexpr float GELU_SQRT_2_OVER_PI = 0.7978845608028654f;
static constexpr float GELU_COEFF = 0.044715f;

__global__ void gelu_fp32_vec4_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n)
{
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
__global__ void gelu_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = v * 0.5f * (1.0f + tanhf(GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v)));
    }
}

// --------------------------------------------------------------------------
// GELU FP16 kernel (load half2, compute in float, store half2)
// --------------------------------------------------------------------------
__global__ void gelu_fp16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int64_t n)
{
    const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        float2 vf = __half22float2(*reinterpret_cast<const __half2*>(x + idx));

        float o0 = vf.x * 0.5f * (1.0f + tanhf(GELU_SQRT_2_OVER_PI * (vf.x + GELU_COEFF * vf.x * vf.x * vf.x)));
        float o1 = vf.y * 0.5f * (1.0f + tanhf(GELU_SQRT_2_OVER_PI * (vf.y + GELU_COEFF * vf.y * vf.y * vf.y)));

        *reinterpret_cast<__half2*>(out + idx) = __float22half2_rn(make_float2(o0, o1));
    } else if (idx < n) {
        float v = __half2float(x[idx]);
        out[idx] = __float2half(v * 0.5f * (1.0f + tanhf(GELU_SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v))));
    }
}

// --------------------------------------------------------------------------
// Host dispatch: swiglu
// --------------------------------------------------------------------------
void swiglu(const Tensor& gate, const Tensor& up, Tensor& out,
            cudaStream_t stream)
{
    const int64_t n = gate.numel();
    if (n == 0) return;

    const int block = 256;

    switch (gate.dtype) {
        case DType::FP32: {
            if (n % 4 == 0 && n >= 4) {
                const int grid = static_cast<int>((n / 4 + block - 1) / block);
                swiglu_fp32_vec4_kernel<<<grid, block, 0, stream>>>(
                    static_cast<const float*>(gate.data),
                    static_cast<const float*>(up.data),
                    static_cast<float*>(out.data),
                    n);
            } else {
                const int grid = static_cast<int>((n + block - 1) / block);
                swiglu_fp32_kernel<<<grid, block, 0, stream>>>(
                    static_cast<const float*>(gate.data),
                    static_cast<const float*>(up.data),
                    static_cast<float*>(out.data),
                    n);
            }
            break;
        }
        case DType::FP16: {
            // Each thread handles 2 elements
            const int64_t half_n = (n + 1) / 2;
            const int grid = static_cast<int>((half_n + block - 1) / block);
            pdl::launch(swiglu_fp16_kernel, dim3(grid), dim3(block), size_t(0), stream,
                static_cast<const __half*>(gate.data),
                static_cast<const __half*>(up.data),
                static_cast<__half*>(out.data),
                n);
            break;
        }
        default:
            break;
    }
}

// --------------------------------------------------------------------------
// Host dispatch: geglu
// --------------------------------------------------------------------------
void geglu(const Tensor& gate, const Tensor& up, Tensor& out,
           cudaStream_t stream)
{
    const int64_t n = gate.numel();
    if (n == 0) return;

    const int block = 256;

    switch (gate.dtype) {
        case DType::FP32: {
            const int grid = static_cast<int>((n + block - 1) / block);
            geglu_fp32_kernel<<<grid, block, 0, stream>>>(
                static_cast<const float*>(gate.data),
                static_cast<const float*>(up.data),
                static_cast<float*>(out.data),
                n);
            break;
        }
        case DType::FP16: {
            const int64_t half_n = (n + 1) / 2;
            const int grid = static_cast<int>((half_n + block - 1) / block);
            pdl::launch(geglu_fp16_kernel, dim3(grid), dim3(block), size_t(0), stream,
                static_cast<const __half*>(gate.data),
                static_cast<const __half*>(up.data),
                static_cast<__half*>(out.data),
                n);
            break;
        }
        default:
            break;
    }
}

// --------------------------------------------------------------------------
// Host dispatch: gelu
// --------------------------------------------------------------------------
void gelu(const Tensor& x, Tensor& out, cudaStream_t stream)
{
    const int64_t n = x.numel();
    if (n == 0) return;

    const int block = 256;

    switch (x.dtype) {
        case DType::FP32: {
            if (n % 4 == 0 && n >= 4) {
                const int grid = static_cast<int>((n / 4 + block - 1) / block);
                gelu_fp32_vec4_kernel<<<grid, block, 0, stream>>>(
                    static_cast<const float*>(x.data),
                    static_cast<float*>(out.data),
                    n);
            } else {
                const int grid = static_cast<int>((n + block - 1) / block);
                gelu_fp32_kernel<<<grid, block, 0, stream>>>(
                    static_cast<const float*>(x.data),
                    static_cast<float*>(out.data),
                    n);
            }
            break;
        }
        case DType::FP16: {
            const int64_t half_n = (n + 1) / 2;
            const int grid = static_cast<int>((half_n + block - 1) / block);
            gelu_fp16_kernel<<<grid, block, 0, stream>>>(
                static_cast<const __half*>(x.data),
                static_cast<__half*>(out.data),
                n);
            break;
        }
        default:
            break;
    }
}

// --------------------------------------------------------------------------
// PDL registration
// --------------------------------------------------------------------------
void activation_pdl_register() {
    pdl::enable(reinterpret_cast<const void*>(&swiglu_fp16_kernel));
    pdl::enable(reinterpret_cast<const void*>(&geglu_fp16_kernel));
}

} // namespace imp
