#include "audio/gemma4_audio_kernels.h"

#include "core/logging.h"  // IMP_CUDA_CHECK_LAUNCH

namespace imp {

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Block-wide sum, kThreads threads; every thread gets the total.
__device__ float block_sum(float v) {
    __shared__ float part[kThreads / 32];
    __shared__ float total;
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0)
        part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float s = threadIdx.x < (kThreads / 32) ? part[threadIdx.x] : 0.0f;
        s = warp_sum(s);
        if (threadIdx.x == 0)
            total = s;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();  // part/total reusable by the next call
    return r;
}

__device__ __forceinline__ float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
__device__ __forceinline__ float silu(float v) { return v / (1.0f + __expf(-v)); }

__global__ void conv2d_s2_kernel(const float* __restrict__ in, const float* __restrict__ w,
                                 float* __restrict__ out, int c_in, int c_out, int rows, int cols, int orows,
                                 int ocols) {
    const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const int64_t total = static_cast<int64_t>(c_out) * orows * ocols;
    if (idx >= total)
        return;
    const int oc = static_cast<int>(idx % ocols);
    const int orow = static_cast<int>((idx / ocols) % orows);
    const int co = static_cast<int>(idx / (static_cast<int64_t>(ocols) * orows));
    float acc = 0.0f;
    for (int ci = 0; ci < c_in; ++ci) {
        const float* plane = in + (static_cast<int64_t>(ci) * rows * cols);
        const float* wk = w + (((static_cast<int64_t>(co) * c_in) + ci) * 9);
        for (int dr = 0; dr < 3; ++dr) {
            const int r = (2 * orow) - 1 + dr;
            if (r < 0 || r >= rows)
                continue;
            for (int dc = 0; dc < 3; ++dc) {
                const int c = (2 * oc) - 1 + dc;
                if (c >= 0 && c < cols)
                    acc += wk[(dr * 3) + dc] * plane[(static_cast<int64_t>(r) * cols) + c];
            }
        }
    }
    out[idx] = acc;
}

__global__ void chan_ln_relu_kernel(const float* __restrict__ x, const float* __restrict__ w, int c, int rows,
                                    int cols, float eps, const uint8_t* __restrict__ row_mask, float* out_chw,
                                    half* out_rc16) {
    const int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const int64_t plane = static_cast<int64_t>(rows) * cols;
    if (idx >= plane)
        return;
    float mean = 0.0f;
    for (int ch = 0; ch < c; ++ch)
        mean += x[(ch * plane) + idx];
    mean /= static_cast<float>(c);
    float var = 0.0f;
    for (int ch = 0; ch < c; ++ch) {
        const float d = x[(ch * plane) + idx] - mean;
        var += d * d;
    }
    const float rstd = rsqrtf((var / static_cast<float>(c)) + eps);
    const int row = static_cast<int>(idx / cols);
    const int col = static_cast<int>(idx % cols);
    const float m = (row_mask != nullptr && row_mask[row] == 0) ? 0.0f : 1.0f;
    for (int ch = 0; ch < c; ++ch) {
        const float y = fmaxf((x[(ch * plane) + idx] - mean) * rstd * w[ch], 0.0f);
        if (out_chw != nullptr)
            out_chw[(ch * plane) + idx] = y * m;
        if (out_rc16 != nullptr)
            out_rc16[(static_cast<int64_t>(row) * cols * c) + (static_cast<int64_t>(col) * c) + ch] =
                __float2half(y);
    }
}

__global__ void rmsnorm_kernel(G4aNormArgs a, int dim) {
    const int64_t base = static_cast<int64_t>(blockIdx.x) * dim;
    auto load = [&](int i) {
        const float v = a.in32 != nullptr ? a.in32[base + i] : __half2float(a.in16[base + i]);
        return clampf(v, -a.clip, a.clip);
    };
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = load(i);
        ss += v * v;
    }
    ss = block_sum(ss);
    const float r = rsqrtf((ss / static_cast<float>(dim)) + a.eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float y = load(i) * r * (a.w != nullptr ? a.w[i] : 1.0f) * a.scale;
        if (a.residual != nullptr)
            y += a.residual[base + i];
        if (a.out16 != nullptr)
            a.out16[base + i] = __float2half(y);
        if (a.out32 != nullptr)
            a.out32[base + i] = y;
    }
}

__global__ void clamp16_kernel(const half* in, half* out, int64_t n, float lo, float hi) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i < n)
        out[i] = __float2half(clampf(__half2float(in[i]), lo, hi));
}

__global__ void silu16_kernel(half* x, int64_t n) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i < n)
        x[i] = __float2half(silu(__half2float(x[i])));
}

__global__ void glu16_kernel(const half* in, half* out, int rows, int d) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= static_cast<int64_t>(rows) * d)
        return;
    const int64_t r = i / d;
    const int64_t c = i % d;
    const float a = __half2float(in[(r * 2 * d) + c]);
    const float b = __half2float(in[(r * 2 * d) + d + c]);
    out[i] = __float2half(a / (1.0f + __expf(-b)));
}

__global__ void add16_to32_kernel(float* x, const half* y, int64_t n) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i < n)
        x[i] += __half2float(y[i]);
}

__global__ void f32_to_16_kernel(const float* x, half* y, int64_t n) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i < n)
        y[i] = __float2half(x[i]);
}

__global__ void add_bias16_kernel(half* x, const float* bias, int rows, int dim) {
    const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i < static_cast<int64_t>(rows) * dim)
        x[i] = __float2half(__half2float(x[i]) + bias[i % dim]);
}

__global__ void dwconv_norm_silu_kernel(const half* __restrict__ in, const float* __restrict__ w_conv,
                                        const float* __restrict__ w_norm, half* __restrict__ out, int d,
                                        int k, float clip, float eps) {
    extern __shared__ float conv[];
    const int row = blockIdx.x;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < k; ++j) {
            const int r = row - (k - 1) + j;
            if (r >= 0)
                acc += w_conv[(c * k) + j] * __half2float(in[(static_cast<int64_t>(r) * d) + c]);
        }
        acc = clampf(acc, -clip, clip);
        conv[c] = acc;
        ss += acc * acc;
    }
    ss = block_sum(ss);
    const float rn = rsqrtf((ss / static_cast<float>(d)) + eps);
    for (int c = threadIdx.x; c < d; c += blockDim.x)
        out[(static_cast<int64_t>(row) * d) + c] = __float2half(silu(conv[c] * rn * w_norm[c]));
}

constexpr int kMaxHdPerLane = 8;  // head_dim <= 256
constexpr int kMaxWindow = 64;

__global__ void local_attention_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                       const half* __restrict__ v, const float* __restrict__ rel_k,
                                       const float* __restrict__ q_scale,
                                       const uint8_t* __restrict__ kv_valid, half* __restrict__ out,
                                       int heads, int hd, int window, int n_rel, float k_scale, float cap) {
    // One warp per (query, head); online softmax over the <= window keys, no per-key frame.
    const int i = static_cast<int>(blockIdx.x);
    const int64_t width = static_cast<int64_t>(heads) * hd;
    const int64_t head_off = static_cast<int64_t>(blockIdx.y) * hd;
    const int lane = static_cast<int>(threadIdx.x);
    const int per = hd / 32;
    const half* qrow = q + (i * width) + head_off;
    float qv[kMaxHdPerLane];
#pragma unroll
    for (int e = 0; e < kMaxHdPerLane; ++e) {
        if (e >= per)
            break;
        const int dd = lane + (e * 32);
        qv[e] = __half2float(qrow[dd]) * q_scale[dd];
    }
    float mx = -3.0e38f;
    float denom = 0.0f;
    float acc[kMaxHdPerLane] = {};
    for (int j = max(0, i - window + 1); j <= i; ++j) {
        if (kv_valid[j] == 0)
            continue;
        const float* rk = rel_k + ((n_rel - 1 - (i - j)) * width) + head_off;
        const half* krow = k + (j * width) + head_off;
        const half* vrow = v + (j * width) + head_off;
        float dot = 0.0f;
#pragma unroll
        for (int e = 0; e < kMaxHdPerLane; ++e) {
            if (e >= per)
                break;
            const int dd = lane + (e * 32);
            dot += qv[e] * ((__half2float(krow[dd]) * k_scale) + rk[dd]);
        }
        dot = warp_sum(dot);
        const float s = cap * tanhf(dot / cap);
        const float m_new = fmaxf(mx, s);
        const float corr = __expf(mx - m_new);
        const float p = __expf(s - m_new);
        denom = (denom * corr) + p;
#pragma unroll
        for (int e = 0; e < kMaxHdPerLane; ++e)
            if (e < per)
                acc[e] = (acc[e] * corr) + (p * __half2float(vrow[lane + (e * 32)]));
        mx = m_new;
    }
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    half* orow = out + (i * width) + head_off;
#pragma unroll
    for (int e = 0; e < kMaxHdPerLane; ++e)
        if (e < per)
            orow[lane + (e * 32)] = __float2half(acc[e] * inv);
}

unsigned grid_for(int64_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

}  // namespace

void launch_g4a_conv2d_s2(const float* in, const float* w, float* out, int c_in, int c_out, int rows,
                          int cols, cudaStream_t stream) {
    const int orows = ((rows - 1) / 2) + 1;
    const int ocols = ((cols - 1) / 2) + 1;
    const int64_t n = static_cast<int64_t>(c_out) * orows * ocols;
    conv2d_s2_kernel<<<grid_for(n), kThreads, 0, stream>>>(in, w, out, c_in, c_out, rows, cols, orows, ocols);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_g4a_chan_ln_relu(const float* x, const float* w, int c, int rows, int cols, float eps,
                             const uint8_t* row_mask, float* out_chw, half* out_rc16, cudaStream_t stream) {
    const int64_t n = static_cast<int64_t>(rows) * cols;
    chan_ln_relu_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, w, c, rows, cols, eps, row_mask, out_chw,
                                                              out_rc16);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_g4a_rmsnorm(const G4aNormArgs& a, int rows, int dim, cudaStream_t stream) {
    rmsnorm_kernel<<<rows, kThreads, 0, stream>>>(a, dim);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_g4a_clamp16(const half* in, half* out, int64_t n, float lo, float hi, cudaStream_t stream) {
    clamp16_kernel<<<grid_for(n), kThreads, 0, stream>>>(in, out, n, lo, hi);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_g4a_silu16(half* x, int64_t n, cudaStream_t stream) {
    silu16_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, n);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_g4a_glu16(const half* in, half* out, int rows, int d, cudaStream_t stream) {
    glu16_kernel<<<grid_for(static_cast<int64_t>(rows) * d), kThreads, 0, stream>>>(in, out, rows, d);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_g4a_add16_to32(float* x, const half* y, int64_t n, cudaStream_t stream) {
    add16_to32_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, y, n);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_g4a_f32_to_16(const float* x, half* y, int64_t n, cudaStream_t stream) {
    f32_to_16_kernel<<<grid_for(n), kThreads, 0, stream>>>(x, y, n);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_g4a_add_bias16(half* x, const float* bias, int rows, int dim, cudaStream_t stream) {
    add_bias16_kernel<<<grid_for(static_cast<int64_t>(rows) * dim), kThreads, 0, stream>>>(x, bias, rows,
                                                                                           dim);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_g4a_dwconv_norm_silu(const half* in, const float* w_conv, const float* w_norm, half* out,
                                 int rows, int d, int k, float clip, float eps, cudaStream_t stream) {
    dwconv_norm_silu_kernel<<<rows, kThreads, static_cast<size_t>(d) * sizeof(float), stream>>>(in, w_conv,
                                                                                                w_norm, out,
                                                                                                d, k, clip,
                                                                                                eps);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_g4a_local_attention(const half* q, const half* k, const half* v, const float* rel_k,
                                const float* q_scale, const uint8_t* kv_valid, half* out, int rows, int heads,
                                int hd, int window, int n_rel, float k_scale, float cap,
                                cudaStream_t stream) {
    if (hd % 32 != 0 || hd > 32 * kMaxHdPerLane || window > kMaxWindow || window > n_rel)
        return;  // config parser refuses these geometries
    local_attention_kernel<<<dim3(static_cast<unsigned>(rows), static_cast<unsigned>(heads)), 32, 0,
                             stream>>>(q, k, v, rel_k, q_scale, kv_valid, out, heads, hd, window, n_rel,
                                       k_scale, cap);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
