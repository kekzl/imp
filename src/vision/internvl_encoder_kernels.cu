#include "vision/internvl_encoder_kernels.h"

#include "core/logging.h"  // IMP_CUDA_CHECK_LAUNCH

namespace imp {

namespace {

constexpr int kBlock = 256;

__global__ void cls_pos_kernel(half* __restrict__ hidden, const half* __restrict__ cls,
                               const half* __restrict__ pos, int64_t total, int dim) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    const int64_t row = i / dim;
    const float base = row == 0 ? __half2float(cls[i % dim]) : __half2float(hidden[i]);
    hidden[i] = __float2half(base + __half2float(pos[i]));
}

__global__ void split_heads_kernel(const half* __restrict__ qkv, half* __restrict__ q, half* __restrict__ k,
                                   half* __restrict__ v, int tokens, int heads, int head_dim) {
    const int64_t width = static_cast<int64_t>(heads) * head_dim;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= tokens * width)
        return;
    const int64_t t = i / width;
    const int64_t c = i % width;
    const int64_t h = c / head_dim, d = c % head_dim;
    const int64_t dst = (h * tokens + t) * head_dim + d;
    const half* src = qkv + t * 3 * width + c;
    q[dst] = src[0];
    k[dst] = src[width];
    v[dst] = src[2 * width];
}

__global__ void scaled_residual_kernel(half* __restrict__ h, const half* __restrict__ x,
                                       const half* __restrict__ scale, int64_t total, int dim) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    h[i] = __float2half(__half2float(h[i]) + __half2float(scale[i % dim]) * __half2float(x[i]));
}

__global__ void pixel_shuffle_kernel(const half* __restrict__ hidden, half* __restrict__ out, int side,
                                     int dim) {
    const int half_side = side / 2;
    const int64_t wide = 4LL * dim;
    const int64_t total = static_cast<int64_t>(half_side) * half_side * wide;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int64_t tok = idx / wide;
    const int64_t c = idx % wide;
    const int64_t i = tok / half_side, j = tok % half_side;
    const int64_t quad = c / dim, ch = c % dim;
    const int64_t r = 2 * i + quad / 2, col = 2 * j + quad % 2;
    out[idx] = hidden[(1 + r * side + col) * dim + ch];  // +1: skip the CLS row
}

__global__ void fill_rows_kernel(half* __restrict__ out, const half* __restrict__ row, int64_t total, int dim) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total)
        out[i] = row[i % dim];
}

unsigned blocks_for(int64_t n) { return static_cast<unsigned>((n + kBlock - 1) / kBlock); }

}  // namespace

void launch_internvl_fill_rows(half* out, const half* row, int rows, int dim, cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(rows) * dim;
    fill_rows_kernel<<<blocks_for(total), kBlock, 0, stream>>>(out, row, total, dim);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_internvl_cls_pos(half* hidden, const half* cls, const half* pos, int tokens, int dim,
                             cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(tokens) * dim;
    cls_pos_kernel<<<blocks_for(total), kBlock, 0, stream>>>(hidden, cls, pos, total, dim);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_internvl_split_heads(const half* qkv, half* q, half* k, half* v, int tokens, int heads,
                                 int head_dim, cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(tokens) * heads * head_dim;
    split_heads_kernel<<<blocks_for(total), kBlock, 0, stream>>>(qkv, q, k, v, tokens, heads, head_dim);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_internvl_scaled_residual(half* h, const half* x, const half* scale, int tokens, int dim,
                                     cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(tokens) * dim;
    scaled_residual_kernel<<<blocks_for(total), kBlock, 0, stream>>>(h, x, scale, total, dim);
    IMP_CUDA_CHECK_LAUNCH();
}

void launch_internvl_pixel_shuffle(const half* hidden, half* out, int side, int dim, cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(side / 2) * (side / 2) * 4 * dim;
    pixel_shuffle_kernel<<<blocks_for(total), kBlock, 0, stream>>>(hidden, out, side, dim);
    IMP_CUDA_CHECK_LAUNCH();
}

}  // namespace imp
