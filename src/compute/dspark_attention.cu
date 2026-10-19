#include "compute/dspark_attention.h"

#include "core/logging.h"

#include <algorithm>
#include <cfloat>

namespace imp {

namespace {

constexpr int kHd = 128;
constexpr int kTile = 32;           // keys per smem tile = lanes
constexpr int kKRow = kHd + 8;      // padded K row (halves): lane-per-key reads stay off one bank
constexpr int kKeysPerSplit = 256;  // split-K granularity
constexpr int kMaxSplits = 64;

__host__ __device__ inline int n_splits_for(int kv_len) {
    const int s = (kv_len + kKeysPerSplit - 1) / kKeysPerSplit;
    return s < 1 ? 1 : (s > kMaxSplits ? kMaxSplits : s);
}

// Keys this launch attends: *d_ctx + rows when the context length lives on the device (graph replay).
__device__ inline int kv_len_of(const int* d_ctx, int rows, int kv_len) {
    return d_ctx ? *d_ctx + rows : kv_len;
}

// grid (n_kv_heads, >= splits), block group * 32: warp w serves query head kv_head * group + w for every row.
// work: o [gridDim.y][n_heads][rows][128], then m and l [gridDim.y][n_heads][rows]; CTAs past the
// kv_len's split count exit.
__global__ void dspark_attn_split_kernel(const half* __restrict__ q, const half* __restrict__ k,
                                         const half* __restrict__ v, float* __restrict__ work, int rows,
                                         int n_heads, int n_kv_heads, const int* __restrict__ d_ctx,
                                         int kv_len_arg, float scale) {
    const int kv_len = kv_len_of(d_ctx, rows, kv_len_arg);
    const int n_splits = n_splits_for(kv_len);
    if (static_cast<int>(blockIdx.y) >= n_splits)
        return;
    const int keys_per_split = (kv_len + n_splits - 1) / n_splits;
    __shared__ __align__(16) half sq[8][kDSparkMaxRows][kHd];
    __shared__ __align__(16) half sk[kTile][kKRow];
    __shared__ __align__(16) half sv[kTile][kHd];
    const int g = blockIdx.x;
    const int split = blockIdx.y;
    const int group = n_heads / n_kv_heads;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int head = (g * group) + warp;
    const int q_stride = n_heads * kHd;
    const int kv_stride = n_kv_heads * kHd;
    for (int i = threadIdx.x; i < group * kDSparkMaxRows * kHd; i += blockDim.x) {
        const int w = i / (kDSparkMaxRows * kHd);
        const int r = (i / kHd) % kDSparkMaxRows;
        const int d = i % kHd;
        sq[w][r][d] =
            r < rows
                ? q[(static_cast<size_t>(r) * q_stride) + (static_cast<size_t>((g * group) + w) * kHd) + d]
                : __float2half(0.f);
    }
    const int j_begin = split * keys_per_split;
    const int j_end = min(kv_len, j_begin + keys_per_split);
    float m[kDSparkMaxRows], l[kDSparkMaxRows], o[kDSparkMaxRows][4];
#pragma unroll
    for (int r = 0; r < kDSparkMaxRows; ++r) {
        m[r] = -FLT_MAX;
        l[r] = 0.f;
        o[r][0] = o[r][1] = o[r][2] = o[r][3] = 0.f;
    }
    for (int j0 = j_begin; j0 < j_end; j0 += kTile) {
        __syncthreads();  // previous tile consumed (and sq written on the first pass)
        for (int i = threadIdx.x; i < kTile * (kHd / 8); i += blockDim.x) {
            const int key = i / (kHd / 8);
            const int c = (i % (kHd / 8)) * 8;
            const bool ok = j0 + key < j_end;
            const size_t src = (static_cast<size_t>(j0 + key) * kv_stride) + (static_cast<size_t>(g) * kHd) +
                               c;
            const uint4 zero = {0u, 0u, 0u, 0u};
            *reinterpret_cast<uint4*>(&sk[key][c]) = ok ? *reinterpret_cast<const uint4*>(k + src) : zero;
            *reinterpret_cast<uint4*>(&sv[key][c]) = ok ? *reinterpret_cast<const uint4*>(v + src) : zero;
        }
        __syncthreads();
        // Scores: lane = key.
        float s[kDSparkMaxRows];
#pragma unroll
        for (int r = 0; r < kDSparkMaxRows; ++r)
            s[r] = 0.f;
        for (int d = 0; d < kHd; d += 2) {
            const float2 kk = __half22float2(*reinterpret_cast<const half2*>(&sk[lane][d]));
#pragma unroll
            for (int r = 0; r < kDSparkMaxRows; ++r) {
                const float2 qq = __half22float2(*reinterpret_cast<const half2*>(&sq[warp][r][d]));
                s[r] += (qq.x * kk.x) + (qq.y * kk.y);
            }
        }
        const bool valid = j0 + lane < j_end;
        float p[kDSparkMaxRows];
#pragma unroll
        for (int r = 0; r < kDSparkMaxRows; ++r) {
            const float sr = valid ? s[r] * scale : -FLT_MAX;
            float mt = sr;
            for (int off = 16; off > 0; off >>= 1)
                mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
            const float mn = fmaxf(m[r], mt);
            const float corr = __expf(m[r] - mn);
            p[r] = valid ? __expf(sr - mn) : 0.f;
            float ps = p[r];
            for (int off = 16; off > 0; off >>= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            l[r] = (l[r] * corr) + ps;
            m[r] = mn;
            o[r][0] *= corr;
            o[r][1] *= corr;
            o[r][2] *= corr;
            o[r][3] *= corr;
        }
        // P V: lane owns dims 4 * lane .. 4 * lane + 3.
        for (int kk = 0; kk < kTile; ++kk) {
            const float2 v01 = __half22float2(
                *reinterpret_cast<const half2*>(&sv[kk][static_cast<size_t>(4) * lane]));
            const float2 v23 = __half22float2(*reinterpret_cast<const half2*>(&sv[kk][(4 * lane) + 2]));
#pragma unroll
            for (int r = 0; r < kDSparkMaxRows; ++r) {
                const float pk = __shfl_sync(0xffffffffu, p[r], kk);
                o[r][0] += pk * v01.x;
                o[r][1] += pk * v01.y;
                o[r][2] += pk * v23.x;
                o[r][3] += pk * v23.y;
            }
        }
    }
    const size_t per_split = static_cast<size_t>(n_heads) * rows;
    float* wo = work + (((static_cast<size_t>(split) * n_heads) + head) * rows * kHd);
    float* wm = work + (static_cast<size_t>(gridDim.y) * per_split * kHd);
    float* wl = wm + (static_cast<size_t>(gridDim.y) * per_split);
#pragma unroll
    for (int r = 0; r < kDSparkMaxRows; ++r) {
        if (r >= rows)
            break;
        float4 out = {o[r][0], o[r][1], o[r][2], o[r][3]};
        *reinterpret_cast<float4*>(wo + (static_cast<size_t>(r) * kHd) +
                                   (static_cast<size_t>(4) * lane)) = out;
        if (lane == 0) {
            const size_t idx = (static_cast<size_t>(split) * per_split) + (static_cast<size_t>(head) * rows) +
                               r;
            wm[idx] = m[r];
            wl[idx] = l[r];
        }
    }
}

// grid (n_heads, rows), block 128 = head_dim: merge the split partials, write FP16.
__global__ void dspark_attn_combine_kernel(const float* __restrict__ work, half* __restrict__ o, int rows,
                                           int n_heads, int grid_splits, const int* __restrict__ d_ctx,
                                           int kv_len_arg) {
    const int head = blockIdx.x;
    const int r = blockIdx.y;
    const int d = threadIdx.x;
    const int splits = n_splits_for(kv_len_of(d_ctx, rows, kv_len_arg));
    const size_t per_split = static_cast<size_t>(n_heads) * rows;
    const float* wm = work + (static_cast<size_t>(grid_splits) * per_split * kHd);
    const float* wl = wm + (static_cast<size_t>(grid_splits) * per_split);
    float mx = -FLT_MAX;
    for (int s = 0; s < splits; ++s)
        mx = fmaxf(mx, wm[(s * per_split) + (static_cast<size_t>(head) * rows) + r]);
    float num = 0.f, den = 0.f;
    for (int s = 0; s < splits; ++s) {
        const size_t idx = (s * per_split) + (static_cast<size_t>(head) * rows) + r;
        const float w = __expf(wm[idx] - mx);
        den += wl[idx] * w;
        num += work[(((static_cast<size_t>(s) * n_heads) + head) * rows * kHd) +
                    (static_cast<size_t>(r) * kHd) + d] *
               w;
    }
    o[(static_cast<size_t>(r) * n_heads * kHd) + (static_cast<size_t>(head) * kHd) + d] = __float2half(num /
                                                                                                       den);
}

}  // namespace

size_t dspark_attention_work_floats(int rows, int n_heads, int max_kv_len) {
    return static_cast<size_t>(n_splits_for(max_kv_len)) * n_heads * rows * (kHd + 2);
}

namespace {

cudaError_t launch_dspark_attention(const half* q, const half* k, const half* v, half* o, float* work,
                                    int rows, int n_heads, int n_kv_heads, const int* d_ctx, int kv_len,
                                    float scale, cudaStream_t stream) {
    if (rows < 1 || rows > kDSparkMaxRows || n_kv_heads < 1 || n_heads % n_kv_heads != 0 ||
        n_heads / n_kv_heads > 8 || kv_len < 1)
        return cudaErrorInvalidValue;
    const int splits = n_splits_for(kv_len);
    dspark_attn_split_kernel<<<dim3(n_kv_heads, splits), (n_heads / n_kv_heads) * 32, 0, stream>>>(
        q, k, v, work, rows, n_heads, n_kv_heads, d_ctx, kv_len, scale);
    IMP_CUDA_CHECK_LAUNCH();
    dspark_attn_combine_kernel<<<dim3(n_heads, rows), kHd, 0, stream>>>(work, o, rows, n_heads, splits, d_ctx,
                                                                        kv_len);
    IMP_CUDA_CHECK_LAUNCH();
    return cudaGetLastError();
}

}  // namespace

cudaError_t dspark_attention(const half* q, const half* k, const half* v, half* o, float* work, int rows,
                             int n_heads, int n_kv_heads, int kv_len, float scale, cudaStream_t stream) {
    return launch_dspark_attention(q, k, v, o, work, rows, n_heads, n_kv_heads, nullptr, kv_len, scale,
                                   stream);
}

cudaError_t dspark_attention_dev(const half* q, const half* k, const half* v, half* o, float* work, int rows,
                                 int n_heads, int n_kv_heads, const int* d_ctx, int max_kv_len, float scale,
                                 cudaStream_t stream) {
    if (!d_ctx)
        return cudaErrorInvalidValue;
    return launch_dspark_attention(q, k, v, o, work, rows, n_heads, n_kv_heads, d_ctx, max_kv_len, scale,
                                   stream);
}

}  // namespace imp
