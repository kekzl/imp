// The drafters' shared norm primitives (audit D-14): head_norm_rope_fp16 (DSpark, per-row positions,
// fp16 weights) and its _seq form (Inkling's draft, pos0 + row, fp32 weights) are one kernel; rms_norm_rows
// serves Inkling (fp32 weights) and DSpark's fc norm (fp16 weights). Equal inputs give equal bits.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "compute/head_norm_rope.h"
#include "compute/rms_norm_rows.h"

using namespace imp;

namespace {

uint32_t g_seed = 12345u;
float rnd() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return static_cast<float>(g_seed >> 8) / 16777216.0f * 2.0f - 1.0f;
}

template <typename T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, h.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return p;
}

template <typename T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

bool same_bits(const void* a, const void* b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; }

// The 3 kernels B-119 replaced, verbatim from 86e5cf30c^ (inkling_draft.cu draft_norm_rope_k, inkling_ops.cu
// rms_norm_k + inkling_detail.cuh block_sum, dspark_drafter.cu dspark_rmsnorm_f32_in_kernel): the oracle.
constexpr int kHd = 128, kThreads = 256;

__global__ void old_draft_norm_rope_k(half* x, int stride, int col0, int heads, const float* w, int pos0,
                                      float log2_theta, float eps, const int* d_pos) {
    __shared__ float part[2];
    const int row = blockIdx.x / heads, head = blockIdx.x % heads;
    half* v = x + (size_t)row * stride + col0 + (size_t)head * kHd;
    const int i = threadIdx.x;
    float a = __half2float(v[i]), b = __half2float(v[i + kHd / 2]);
    if (w) {
        float ss = a * a + b * b;
        for (int o = 16; o > 0; o >>= 1)
            ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if ((i & 31) == 0)
            part[i >> 5] = ss;
        __syncthreads();
        const float inv = rsqrtf((part[0] + part[1]) / kHd + eps);
        a *= inv * w[i];
        b *= inv * w[i + kHd / 2];
    }
    const float freq = exp2f(-log2_theta * (2.0f * i) / kHd);
    float sn = 0.0f, cs = 0.0f;
    sincosf(static_cast<float>(pos0 + (d_pos ? *d_pos : 0) + row) * freq, &sn, &cs);
    v[i] = __float2half(a * cs - b * sn);
    v[i + kHd / 2] = __float2half(b * cs + a * sn);
}

__device__ inline float old_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ inline float old_block_sum(float v, float* sh) {
    v = old_warp_sum(v);
    if ((threadIdx.x & 31) == 0)
        sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i)
        t += sh[i];
    __syncthreads();
    return t;
}

__global__ void old_rms_norm_k(const float* x, const float* w, float* of, half* oh, int H, float eps) {
    __shared__ float sh[kThreads / 32];
    const float* xr = x + (size_t)blockIdx.x * H;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        ss += xr[i] * xr[i];
    const float r = rsqrtf(old_block_sum(ss, sh) / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float v = xr[i] * r * (w ? w[i] : 1.0f);
        if (of)
            of[(size_t)blockIdx.x * H + i] = v;
        if (oh)
            oh[(size_t)blockIdx.x * H + i] = __float2half(v);
    }
}

__global__ void old_dspark_rmsnorm_f32_in_kernel(const float* __restrict__ x, const half* __restrict__ w,
                                                 half* __restrict__ out, int d, float eps) {
    const float* xr = x + (static_cast<size_t>(blockIdx.x) * d);
    float ss = 0.0f;
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        ss += xr[j] * xr[j];
    for (int o = 16; o > 0; o >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    __shared__ float part[8];
    if ((threadIdx.x & 31) == 0)
        part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (int k = 0; k < static_cast<int>(blockDim.x >> 5); ++k)
        tot += part[k];
    const float inv = rsqrtf((tot / static_cast<float>(d)) + eps);
    half* orow = out + (static_cast<size_t>(blockIdx.x) * d);
    for (int j = threadIdx.x; j < d; j += blockDim.x)
        orow[j] = __float2half(xr[j] * inv * __half2float(w[j]));
}

}  // namespace

// Inkling draft shapes (InklingDraftConfig: 24 q / 8 kv heads of 128, theta 1e6, block 8): the context K
// rows (stride 2 * KD, 1024-row segment at pos0 79282) and the block's q and k (stride QD + 2 * KD, col0 0
// and QD, pos0 0 + *d_pos 81000). The former kernel and head_norm_rope_fp16_seq give the same bits.
TEST(DrafterNorms, HeadNormRopeSeqMatchesTheFormerInklingKernel) {
    constexpr int kHeads = 24, kKv = 8, QD = kHeads * kHd, KD = kKv * kHd;
    struct Call {
        int rows, stride, col0, heads, pos0;
        bool d_pos;
    };
    const Call calls[] = {{1024, 2 * KD, 0, kKv, 79282, false},
                          {8, QD + 2 * KD, 0, kHeads, 0, true},
                          {8, QD + 2 * KD, QD, kKv, 0, true}};
    std::vector<float> wf(kHd);
    for (auto& v : wf)
        v = __half2float(__float2half(1.0f + rnd() * 0.5f));
    float* d_wf = dev(wf);
    int* d_base = dev(std::vector<int>{81000});
    const float log2_theta = std::log2(1e6f), eps = 1e-6f;
    for (const Call& c : calls) {
        std::vector<half> x(static_cast<size_t>(c.rows) * c.stride);
        for (auto& v : x)
            v = __float2half(rnd() * 4.0f);
        for (const bool norm : {true, false}) {
            half* xa = dev(x);
            half* xb = dev(x);
            const float* w = norm ? d_wf : nullptr;
            const int* dp = c.d_pos ? d_base : nullptr;
            old_draft_norm_rope_k<<<c.rows * c.heads, kHd / 2>>>(xa, c.stride, c.col0, c.heads, w, c.pos0,
                                                                 log2_theta, eps, dp);
            head_norm_rope_fp16_seq(xb + c.col0, c.rows, c.stride, c.heads, w, c.pos0, dp, log2_theta, eps,
                                    nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const auto a = host(xa, x.size()), b = host(xb, x.size());
            EXPECT_TRUE(same_bits(a.data(), b.data(), a.size() * sizeof(half)))
                << "rows " << c.rows << " col0 " << c.col0 << (norm ? " norm" : " rope only");
            EXPECT_FALSE(same_bits(a.data(), x.data(), a.size() * sizeof(half)));
            cudaFree(xa);
            cudaFree(xb);
        }
    }
    cudaFree(d_wf);
    cudaFree(d_base);
}

// Inkling's rms_norm (hidden 2304: fp32 and / or fp16 out, in place, no weight; 1024-row segment) against
// the former rms_norm_k, and DSpark's fc norm (fp16 weights, fp16 out, d 2304 and 5120) against
// dspark_rmsnorm_f32_in_kernel: the same bits.
TEST(DrafterNorms, RmsNormRowsMatchesTheFormerCopies) {
    const float eps = 1e-6f;
    for (const int H : {2304, 5120}) {
        constexpr int kRows = 1024;
        const size_t n = static_cast<size_t>(kRows) * H;
        std::vector<float> x(n);
        for (auto& v : x)
            v = rnd() * 50.0f;
        std::vector<float> wf(H);
        std::vector<half> wh(H);
        for (int i = 0; i < H; ++i) {
            wh[static_cast<size_t>(i)] = __float2half(0.5f + rnd());
            wf[static_cast<size_t>(i)] = 0.5f + rnd();
        }
        float* d_x = dev(x);
        float* d_wf = dev(wf);
        half* d_wh = dev(wh);
        float *of_a = nullptr, *of_b = nullptr;
        half *oh_a = nullptr, *oh_b = nullptr;
        ASSERT_EQ(cudaMalloc(&of_a, n * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&of_b, n * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&oh_a, n * sizeof(half)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&oh_b, n * sizeof(half)), cudaSuccess);
        auto expect_same = [&](const char* what) {
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            EXPECT_TRUE(same_bits(host(of_a, n).data(), host(of_b, n).data(), n * sizeof(float)))
                << what << " fp32, H " << H;
            EXPECT_TRUE(same_bits(host(oh_a, n).data(), host(oh_b, n).data(), n * sizeof(half)))
                << what << " fp16, H " << H;
        };
        for (const bool weighted : {true, false}) {
            const float* w = weighted ? d_wf : nullptr;
            old_rms_norm_k<<<kRows, kThreads>>>(d_x, w, of_a, oh_a, H, eps);
            rms_norm_rows(d_x, w, of_b, oh_b, kRows, H, eps, nullptr);
            expect_same(weighted ? "inkling" : "inkling unweighted");
        }
        // In place (the embedding norm: out_f aliases x).
        float* xa = dev(x);
        float* xb = dev(x);
        old_rms_norm_k<<<kRows, kThreads>>>(xa, d_wf, xa, nullptr, H, eps);
        rms_norm_rows(xb, d_wf, xb, nullptr, kRows, H, eps, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_TRUE(same_bits(host(xa, n).data(), host(xb, n).data(), n * sizeof(float)))
            << "in place, H " << H;
        old_dspark_rmsnorm_f32_in_kernel<<<kRows, 256>>>(d_x, d_wh, oh_a, H, eps);
        rms_norm_rows(d_x, d_wh, nullptr, oh_b, kRows, H, eps, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_TRUE(same_bits(host(oh_a, n).data(), host(oh_b, n).data(), n * sizeof(half)))
            << "dspark, H " << H;
        for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_wf), static_cast<void*>(d_wh),
                        static_cast<void*>(of_a), static_cast<void*>(of_b), static_cast<void*>(oh_a),
                        static_cast<void*>(oh_b), static_cast<void*>(xa), static_cast<void*>(xb)})
            cudaFree(p);
    }
}

// Inkling's draft calls: positions pos0 + *d_pos0 + row, fp32 weights holding fp16 values; DSpark's: the
// per-row table, the same weights as fp16. Both arms, with and without the norm.
TEST(DrafterNorms, HeadNormRopeSeqEqualsPerRowPositions) {
    constexpr int kRows = 5, kHeads = 3, kStride = kHeads * kHeadNormRopeDim + 64;
    std::vector<half> x(static_cast<size_t>(kRows) * kStride);
    for (auto& v : x)
        v = __float2half(rnd() * 4.0f);
    std::vector<half> wh(kHeadNormRopeDim);
    std::vector<float> wf(kHeadNormRopeDim);
    for (int i = 0; i < kHeadNormRopeDim; ++i) {
        wh[static_cast<size_t>(i)] = __float2half(1.0f + rnd() * 0.5f);
        wf[static_cast<size_t>(i)] = __half2float(wh[static_cast<size_t>(i)]);
    }
    constexpr int kPos0 = 1000, kBase = 37;
    std::vector<int> rows(kRows);
    for (int r = 0; r < kRows; ++r)
        rows[static_cast<size_t>(r)] = kPos0 + kBase + r;
    const float log2_theta = std::log2(1000000.0f), eps = 1e-6f;
    half* xa = dev(x);
    half* xb = dev(x);
    half* d_wh = dev(wh);
    float* d_wf = dev(wf);
    int* d_rows = dev(rows);
    int* d_base = dev(std::vector<int>{kBase});
    for (const bool norm : {true, false}) {
        head_norm_rope_fp16(xa, kRows, kStride, kHeads, norm ? d_wh : nullptr, d_rows, log2_theta, eps,
                            nullptr);
        head_norm_rope_fp16_seq(xb, kRows, kStride, kHeads, norm ? d_wf : nullptr, kPos0, d_base, log2_theta,
                                eps, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto a = host(xa, x.size()), b = host(xb, x.size());
        EXPECT_TRUE(same_bits(a.data(), b.data(), a.size() * sizeof(half))) << (norm ? "norm" : "rope only");
        EXPECT_FALSE(same_bits(a.data(), x.data(), a.size() * sizeof(half)));
    }
    for (void* p : {static_cast<void*>(xa), static_cast<void*>(xb), static_cast<void*>(d_wh),
                    static_cast<void*>(d_wf), static_cast<void*>(d_rows), static_cast<void*>(d_base)})
        cudaFree(p);
}

// fp16 and fp32 weights of the same values give the same bits, in place (Inkling's embedding norm) as
// out of place, and the rows match a double reference.
TEST(DrafterNorms, RmsNormRowsWeightTypesAgreeAndMatchReference) {
    constexpr int kRows = 4, kH = 2304;
    std::vector<float> x(static_cast<size_t>(kRows) * kH);
    for (auto& v : x)
        v = rnd() * 50.0f;
    std::vector<half> wh(kH);
    std::vector<float> wf(kH);
    for (int i = 0; i < kH; ++i) {
        wh[static_cast<size_t>(i)] = __float2half(0.5f + rnd());
        wf[static_cast<size_t>(i)] = __half2float(wh[static_cast<size_t>(i)]);
    }
    const float eps = 1e-6f;
    float* d_x = dev(x);
    float* d_inplace = dev(x);
    half* d_wh = dev(wh);
    float* d_wf = dev(wf);
    float *of_h = nullptr, *of_f = nullptr;
    half *oh_h = nullptr, *oh_f = nullptr;
    ASSERT_EQ(cudaMalloc(&of_h, x.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&of_f, x.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&oh_h, x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&oh_f, x.size() * sizeof(half)), cudaSuccess);
    rms_norm_rows(d_x, d_wh, of_h, oh_h, kRows, kH, eps, nullptr);
    rms_norm_rows(d_x, d_wf, of_f, oh_f, kRows, kH, eps, nullptr);
    rms_norm_rows(d_inplace, d_wf, d_inplace, nullptr, kRows, kH, eps, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto fh = host(of_h, x.size()), ff = host(of_f, x.size()), fi = host(d_inplace, x.size());
    const auto hh = host(oh_h, x.size()), hf = host(oh_f, x.size());
    EXPECT_TRUE(same_bits(fh.data(), ff.data(), fh.size() * sizeof(float)));
    EXPECT_TRUE(same_bits(ff.data(), fi.data(), ff.size() * sizeof(float)));
    EXPECT_TRUE(same_bits(hh.data(), hf.data(), hh.size() * sizeof(half)));
    for (int r = 0; r < kRows; ++r) {
        double ss = 0.0;
        for (int i = 0; i < kH; ++i)
            ss += static_cast<double>(x[static_cast<size_t>(r) * kH + i]) *
                  x[static_cast<size_t>(r) * kH + i];
        const double inv = 1.0 / std::sqrt(ss / kH + eps);
        for (int i = 0; i < kH; i += 97) {
            const size_t k = static_cast<size_t>(r) * kH + i;
            const double ref = x[k] * inv * wf[static_cast<size_t>(i)];
            EXPECT_NEAR(ff[k], ref, 1e-5 * std::fabs(ref) + 1e-6) << "row " << r << " col " << i;
        }
    }
    for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_inplace), static_cast<void*>(d_wh),
                    static_cast<void*>(d_wf), static_cast<void*>(of_h), static_cast<void*>(of_f),
                    static_cast<void*>(oh_h), static_cast<void*>(oh_f)})
        cudaFree(p);
}
