#include "vision/qwen3vl_encoder_kernels.h"

#include <cmath>

namespace imp {

namespace {

constexpr int kBlock = 256;

__global__ void pos_embed_add_kernel(half* __restrict__ hidden, const half* __restrict__ table,
                                     const int32_t* __restrict__ taps, const float* __restrict__ weights,
                                     int dim, int taps_per_token) {
    const int token = blockIdx.x;
    const int32_t* my_taps = taps + static_cast<int64_t>(token) * taps_per_token;
    const float* my_w = weights + static_cast<int64_t>(token) * taps_per_token;
    half* row = hidden + static_cast<int64_t>(token) * dim;

    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float acc = __half2float(row[j]);
        for (int t = 0; t < taps_per_token; ++t) {
            const float w = my_w[t];
            // A zero weight is the common case at the grid edges; skipping the
            // load matters because these are scattered rows.
            if (w != 0.0f)
                acc += w * __half2float(table[static_cast<int64_t>(my_taps[t]) * dim + j]);
        }
        row[j] = __float2half(acc);
    }
}

// One block per row, FP32 accumulation. `dim` is 1024 or 4096 here, so a plain
// two-pass block reduction is both simple and enough.
__global__ void layernorm_kernel(const half* __restrict__ x, const half* __restrict__ weight,
                                 const half* __restrict__ bias, half* __restrict__ out, int dim, float eps) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const half* xr = x + static_cast<int64_t>(row) * dim;
    half* outr = out + static_cast<int64_t>(row) * dim;

    float sum = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        sum += __half2float(xr[j]);
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float mean = smem[0] / dim;
    __syncthreads();

    float var = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        const float d = __half2float(xr[j]) - mean;
        var += d * d;
    }
    smem[threadIdx.x] = var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float inv_std = rsqrtf(smem[0] / dim + eps);

    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        const float v = (__half2float(xr[j]) - mean) * inv_std;
        outr[j] = __float2half(v * __half2float(weight[j]) + __half2float(bias[j]));
    }
}

__global__ void add_bias_kernel(half* __restrict__ x, const half* __restrict__ bias, int64_t n, int dim) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = __float2half(__half2float(x[i]) + __half2float(bias[i % dim]));
}

__global__ void residual_add_kernel(half* __restrict__ dst, const half* __restrict__ src, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

__global__ void gelu_tanh_kernel(half* __restrict__ x, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float v = __half2float(x[i]);
    const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    x[i] = __float2half(0.5f * v * (1.0f + tanhf(inner)));
}

__global__ void gelu_erf_kernel(half* __restrict__ x, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float v = __half2float(x[i]);
    x[i] = __float2half(0.5f * v * (1.0f + erff(v * 0.7071067811865476f)));
}

// One block per (head, token). Reads the fused row once, writes q/k/v into the
// per-head layout the batched attention GEMMs want, and rotates q/k on the way.
__global__ void split_qkv_rope_kernel(const half* __restrict__ qkv, const int32_t* __restrict__ row_id,
                                      const int32_t* __restrict__ col_id, half* __restrict__ q,
                                      half* __restrict__ k, half* __restrict__ v, int tokens, int heads,
                                      int head_dim, float theta) {
    const int head = blockIdx.y;
    const int token = blockIdx.x;
    const int hidden = heads * head_dim;
    const int half_rot = head_dim / 2;  // rotated pair distance
    const int quarter = half_rot / 2;   // where the row axis hands over to the column axis
    const int64_t src = static_cast<int64_t>(token) * 3 * hidden + head * head_dim;
    const int64_t dst = (static_cast<int64_t>(head) * tokens + token) * head_dim;

    for (int j = threadIdx.x; j < head_dim; j += blockDim.x)
        v[dst + j] = qkv[src + 2 * hidden + j];

    const float r = static_cast<float>(row_id[token]);
    const float c = static_cast<float>(col_id[token]);
    for (int j = threadIdx.x; j < half_rot; j += blockDim.x) {
        // The rotary index j runs over head_dim/2 angles; the first quarter of
        // head_dim is driven by the row, the second by the column.
        const int fi = (j < quarter) ? j : (j - quarter);
        const float pos = (j < quarter) ? r : c;
        const float inv_freq = __powf(theta, -static_cast<float>(2 * fi) / static_cast<float>(half_rot));
        const float ang = pos * inv_freq;
        float cs, sn;
        __sincosf(ang, &sn, &cs);

        const float q0 = __half2float(qkv[src + j]);
        const float q1 = __half2float(qkv[src + j + half_rot]);
        q[dst + j] = __float2half(q0 * cs - q1 * sn);
        q[dst + j + half_rot] = __float2half(q1 * cs + q0 * sn);

        const float k0 = __half2float(qkv[src + hidden + j]);
        const float k1 = __half2float(qkv[src + hidden + j + half_rot]);
        k[dst + j] = __float2half(k0 * cs - k1 * sn);
        k[dst + j + half_rot] = __float2half(k1 * cs + k0 * sn);
    }
}

__global__ void softmax_rows_kernel(half* __restrict__ scores, int cols) {
    extern __shared__ float smem[];
    const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;

    float m = -INFINITY;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        m = fmaxf(m, __half2float(scores[base + j]));
    smem[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    const float row_max = smem[0];
    __syncthreads();

    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        const float e = __expf(__half2float(scores[base + j]) - row_max);
        scores[base + j] = __float2half(e);
        sum += e;
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / smem[0];
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        scores[base + j] = __float2half(__half2float(scores[base + j]) * inv);
}

__global__ void merge_heads_kernel(const half* __restrict__ per_head, half* __restrict__ out, int tokens,
                                   int heads, int head_dim) {
    const int token = blockIdx.x;
    const int hidden = heads * head_dim;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        const int head = j / head_dim;
        const int d = j % head_dim;
        out[static_cast<int64_t>(token) * hidden + j] =
            per_head[(static_cast<int64_t>(head) * tokens + token) * head_dim + d];
    }
}

int reduce_threads(int dim) {
    int t = 32;
    while (t < dim && t < 1024)
        t *= 2;
    return t;
}

}  // namespace

void launch_qwen3vl_pos_embed_add(half* hidden, const half* table, const int32_t* taps, const float* weights,
                                  int tokens, int dim, int taps_per_token, cudaStream_t stream) {
    pos_embed_add_kernel<<<tokens, kBlock, 0, stream>>>(hidden, table, taps, weights, dim, taps_per_token);
}

void launch_qwen3vl_layernorm(const half* x, const half* weight, const half* bias, half* out, int rows,
                              int dim, float eps, cudaStream_t stream) {
    const int threads = reduce_threads(dim);
    layernorm_kernel<<<rows, threads, threads * sizeof(float), stream>>>(x, weight, bias, out, dim, eps);
}

void launch_qwen3vl_add_bias(half* x, const half* bias, int rows, int dim, cudaStream_t stream) {
    const int64_t n = static_cast<int64_t>(rows) * dim;
    add_bias_kernel<<<static_cast<int>((n + kBlock - 1) / kBlock), kBlock, 0, stream>>>(x, bias, n, dim);
}

void launch_qwen3vl_residual_add(half* dst, const half* src, int64_t n, cudaStream_t stream) {
    residual_add_kernel<<<static_cast<int>((n + kBlock - 1) / kBlock), kBlock, 0, stream>>>(dst, src, n);
}

void launch_qwen3vl_gelu_tanh(half* x, int64_t n, cudaStream_t stream) {
    gelu_tanh_kernel<<<static_cast<int>((n + kBlock - 1) / kBlock), kBlock, 0, stream>>>(x, n);
}

void launch_qwen3vl_gelu_erf(half* x, int64_t n, cudaStream_t stream) {
    gelu_erf_kernel<<<static_cast<int>((n + kBlock - 1) / kBlock), kBlock, 0, stream>>>(x, n);
}

void launch_qwen3vl_split_qkv_rope(const half* qkv, const int32_t* row, const int32_t* col, half* q, half* k,
                                   half* v, int tokens, int heads, int head_dim, float theta,
                                   cudaStream_t stream) {
    dim3 grid(tokens, heads);
    split_qkv_rope_kernel<<<grid, 64, 0, stream>>>(qkv, row, col, q, k, v, tokens, heads, head_dim, theta);
}

void launch_qwen3vl_softmax_rows(half* scores, int rows, int cols, cudaStream_t stream) {
    const int threads = reduce_threads(cols);
    softmax_rows_kernel<<<rows, threads, threads * sizeof(float), stream>>>(scores, cols);
}

void launch_qwen3vl_merge_heads(const half* per_head, half* out, int tokens, int heads, int head_dim,
                                cudaStream_t stream) {
    merge_heads_kernel<<<tokens, kBlock, 0, stream>>>(per_head, out, tokens, heads, head_dim);
}

}  // namespace imp
