// Mamba2 prefill front end in one launch: causal conv1d + SiLU + x/B/C split.
// Reads the xBC columns straight out of the ssm_in projection rows (any pitch) and writes
// x, B and C as contiguous [n, inner] / [n, BC] planes, the layout the scan takes.
// Bit-identical to ssm_conv1d_prefill + silu_inplace + the three cudaMemcpy2DAsync.
#include "compute/ssm.h"

#include "core/logging.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace imp {
namespace {

constexpr int kVec = 8;        // channels per thread (one uint4 of halves)
constexpr int kThreads = 128;  // 1024 channels per CTA
constexpr int kTile = 32;      // tokens per CTA
constexpr int kMaxK = 4;       // conv kernel size covered

__device__ __forceinline__ void unpack8(const uint4& v, float (&f)[kVec]) {
    const half2* h = reinterpret_cast<const half2*>(&v);
#pragma unroll
    for (int i = 0; i < kVec / 2; ++i) {
        const float2 p = __half22float2(h[i]);
        const int j = 2 * i;
        f[j] = p.x;
        f[j + 1] = p.y;
    }
}

// Window row r (r = token - (K-1) + k) of channels ch0..ch0+7: input rows from the
// projection, rows before the chunk from conv_state[ch * K + (r + K)].
template <int K>
__device__ __forceinline__ void load_row(float (&f)[kVec], const half* __restrict__ x_in, int64_t pitch,
                                         const float* __restrict__ conv_state, int r, int ch0) {
    if (r >= 0) {
        unpack8(__ldg(reinterpret_cast<const uint4*>(x_in + (static_cast<int64_t>(r) * pitch) + ch0)), f);
    } else {
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            const int s = r + K;
            f[i] = (conv_state && s >= 0) ? conv_state[((ch0 + i) * K) + s] : 0.0f;
        }
    }
}

template <int K>
__global__ void __launch_bounds__(kThreads) conv_silu_split_kernel(
    const float* __restrict__ conv_state, const half* __restrict__ x_in, int64_t pitch,
    const half* __restrict__ weight, const half* __restrict__ bias, half* __restrict__ x_out,
    half* __restrict__ b_out, half* __restrict__ c_out, int n_tokens, int inner, int bc) {
    const int channels = inner + (2 * bc);
    const int ch0 = ((blockIdx.x * kThreads) + threadIdx.x) * kVec;
    if (ch0 >= channels)
        return;
    const int t0 = blockIdx.y * kTile;

    float w[kVec][K];
    float bv[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
#pragma unroll
        for (int k = 0; k < K; ++k)
            w[i][k] = __half2float(weight[((ch0 + i) * K) + k]);
        bv[i] = bias ? __half2float(bias[ch0 + i]) : 0.0f;
    }

    half* dst;
    int64_t dst_pitch;
    if (ch0 < inner) {
        dst = x_out + ch0;
        dst_pitch = inner;
    } else if (ch0 < inner + bc) {
        dst = b_out + (ch0 - inner);
        dst_pitch = bc;
    } else {
        dst = c_out + (ch0 - inner - bc);
        dst_pitch = bc;
    }

    // win[k] holds row t - (K-1) + k.
    float win[K][kVec];
#pragma unroll
    for (int k = 0; k < K - 1; ++k)
        load_row<K>(win[k + 1], x_in, pitch, conv_state, t0 - (K - 1) + k, ch0);
    const int t_end = min(t0 + kTile, n_tokens);
    for (int t = t0; t < t_end; ++t) {
#pragma unroll
        for (int k = 0; k < K - 1; ++k)
#pragma unroll
            for (int i = 0; i < kVec; ++i)
                win[k][i] = win[k + 1][i];
        load_row<K>(win[K - 1], x_in, pitch, conv_state, t, ch0);
        uint4 packed;
        half2* out = reinterpret_cast<half2*>(&packed);
#pragma unroll
        for (int i = 0; i < kVec; i += 2) {
            float v[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                // Same order and roundings as ssm_conv1d_prefill_kernel + silu_inplace_fp16_kernel.
                float sum = 0.0f;
#pragma unroll
                for (int k = 0; k < K; ++k)
                    sum += win[k][i + u] * w[i + u][k];
                if (bias)
                    sum += bv[i + u];
                float val = __half2float(__float2half(sum));
                val = val / (1.0f + expf(-val));
                v[u] = val;
            }
            out[i / 2] = __halves2half2(__float2half(v[0]), __float2half(v[1]));
        }
        *reinterpret_cast<uint4*>(dst + (static_cast<int64_t>(t) * dst_pitch)) = packed;
    }
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

bool ssm_conv1d_prefill_silu_split(const void* conv_state, const half* x_in, int64_t pitch,
                                   const Tensor& weight, const Tensor& bias, half* x_out, half* b_out,
                                   half* c_out, int n_tokens, int inner, int bc, int conv_kernel,
                                   cudaStream_t stream) {
    if (conv_kernel != kMaxK || n_tokens <= 0 || inner % kVec != 0 || bc % kVec != 0 || pitch % kVec != 0 ||
        !aligned16(x_in) || !aligned16(x_out) || !aligned16(b_out) || !aligned16(c_out))
        return false;
    const int channels = inner + (2 * bc);
    const dim3 grid((channels + (kThreads * kVec) - 1) / (kThreads * kVec), (n_tokens + kTile - 1) / kTile);
    conv_silu_split_kernel<kMaxK>
        <<<grid, kThreads, 0, stream>>>(static_cast<const float*>(conv_state), x_in, pitch,
                                        static_cast<const half*>(weight.data),
                                        bias.data ? static_cast<const half*>(bias.data) : nullptr, x_out,
                                        b_out, c_out, n_tokens, inner, bc);
    IMP_CUDA_CHECK_LAUNCH();
    return true;
}

// Prefill: x/B/C columns of the conv output into contiguous planes (x -> x_dst, B then C -> bc_dst).
void ssm_deinterleave_xbc(const Tensor& xBC_out, void* x_dst, void* bc_dst, int n, int inner, int bc,
                          size_t es, cudaStream_t stream) {
    const size_t pitch = static_cast<size_t>(inner + (2 * bc)) * es;
    const char* src = static_cast<const char*>(xBC_out.data);
    char* b_dst = static_cast<char*>(bc_dst);
    char* c_dst = b_dst + (static_cast<size_t>(n) * bc * es);
    IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(x_dst, static_cast<size_t>(inner) * es, src, pitch,
                                         static_cast<size_t>(inner) * es, n, cudaMemcpyDeviceToDevice,
                                         stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(b_dst, static_cast<size_t>(bc) * es, src + (inner * es), pitch,
                                         static_cast<size_t>(bc) * es, n, cudaMemcpyDeviceToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpy2DAsync(c_dst, static_cast<size_t>(bc) * es, src + ((inner + bc) * es),
                                         pitch, static_cast<size_t>(bc) * es, n, cudaMemcpyDeviceToDevice,
                                         stream));
}

}  // namespace imp
