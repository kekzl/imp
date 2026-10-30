// Own grouped NVFP4 MoE GEMM argument forms: host-args (gemm_grouped_nvfp4_host_args) vs a per-expert
// dense CUTLASS NVFP4 GEMM on the same bits, and device-args (gemm_grouped_nvfp4_tile, B slab mode (a),
// pointer arrays mode (b), expert-block slices) bit-identical to host-args.

#include <gtest/gtest.h>
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_grouped_nvfp4_tile.h"
#include "core/tensor.h"
#include "quant/nvfp4_quant.h"
#include "scoped_engine_arena.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>
#include <cmath>
#include <random>

namespace imp {
namespace {

class GroupedNvfp4ArgsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // T2 arena for the host-args staging slice; Engine::init opens it in production.
        arena_ = std::make_unique<ScopedEngineArena>();
        gemm_grouped_nvfp4_prewarm();
        cudaStreamCreate(&stream_);
        int dev = 0;
        cudaGetDevice(&dev);
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        sm_ = major * 10 + minor;
    }
    void TearDown() override {
        cudaStreamDestroy(stream_);
        arena_.reset();
    }
    std::unique_ptr<ScopedEngineArena> arena_;
    cudaStream_t stream_ = nullptr;
    int sm_ = 0;
};

struct SyntheticExpert {
    NvFP4QuantResult nvfp4{};
    CutlassNvFP4Weight cutlass_w{};
};

static void make_expert(SyntheticExpert& e, int N, int K, float wscale, uint64_t seed, cudaStream_t stream) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-wscale, wscale);
    std::vector<half> w(static_cast<size_t>(N) * K);
    for (auto& v : w)
        v = __float2half(dist(gen));
    void* d_w_fp16 = nullptr;
    cudaMalloc(&d_w_fp16, w.size() * sizeof(half));
    cudaMemcpy(d_w_fp16, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice);
    int64_t w_shape[2] = {N, K};
    Tensor w_input(d_w_fp16, QType::F16, 2, w_shape, true);
    quantize_fp16_to_nvfp4(w_input, e.nvfp4, stream);
    convert_nvfp4_to_cutlass(e.nvfp4, e.cutlass_w, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_w_fp16);
}

static void free_expert(SyntheticExpert& e) {
    free_cutlass_nvfp4_weight(e.cutlass_w);
    free_nvfp4_result(e.nvfp4);
}

// Expert-sorted operands as the MoE prefill builds them: contiguous A / SFA / D slabs, B / SFB in one
// slab with a per-expert stride. Experts get distinct weight scales (distinct alpha).
struct Problem {
    int ne = 0, N = 0, K = 0, M_total = 0;
    std::vector<int> M_per, offsets;  // offsets [ne+1]
    std::vector<int64_t> sfa_off;     // [ne+1]
    std::vector<float> alpha;
    std::vector<SyntheticExpert> experts;
    void *a = nullptr, *a_sf = nullptr, *b = nullptr, *b_sf = nullptr;
    size_t b_stride = 0, sfb_stride = 0;

    void build(const std::vector<int>& m_per, int n, int k, uint64_t seed, cudaStream_t s) {
        M_per = m_per;
        ne = static_cast<int>(m_per.size());
        N = n;
        K = k;
        offsets.assign(ne + 1, 0);
        sfa_off.assign(ne + 1, 0);
        for (int e = 0; e < ne; ++e) {
            offsets[e + 1] = offsets[e] + M_per[e];
            sfa_off[e + 1] = sfa_off[e] + static_cast<int64_t>(cutlass_nvfp4_sf_size(M_per[e], K));
        }
        M_total = offsets[ne];
        b_stride = static_cast<size_t>(N) * K / 2;
        sfb_stride = cutlass_nvfp4_sf_size(N, K);
        cudaMalloc(&b, b_stride * ne);
        cudaMalloc(&b_sf, sfb_stride * ne);
        experts.resize(ne);
        alpha.resize(ne);
        for (int e = 0; e < ne; ++e) {
            make_expert(experts[e], N, K, 0.25f * static_cast<float>(1 + e), seed + e, s);
            cudaMemcpyAsync(static_cast<char*>(b) + (e * b_stride), experts[e].cutlass_w.data, b_stride,
                            cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(static_cast<char*>(b_sf) + (e * sfb_stride), experts[e].cutlass_w.scale_factors,
                            sfb_stride, cudaMemcpyDeviceToDevice, s);
            alpha[e] = experts[e].cutlass_w.tensor_scale;
        }
        std::mt19937 agen(static_cast<unsigned>(seed));
        std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
        std::vector<half> h_x(static_cast<size_t>(M_total) * K);
        for (auto& v : h_x)
            v = __float2half(adist(agen));
        void* x = nullptr;
        cudaMalloc(&x, h_x.size() * sizeof(half));
        cudaMemcpy(x, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
        cudaMalloc(&a, static_cast<size_t>(M_total) * K / 2);
        cudaMalloc(&a_sf, sfa_off[ne]);
        cudaMemsetAsync(a_sf, 0, sfa_off[ne], s);
        for (int e = 0; e < ne; ++e)
            if (M_per[e] > 0)
                quantize_fp16_to_nvfp4_cutlass(static_cast<const half*>(x) +
                                                   (static_cast<size_t>(offsets[e]) * K),
                                               static_cast<char*>(a) +
                                                   (static_cast<size_t>(offsets[e]) * K / 2),
                                               static_cast<char*>(a_sf) + sfa_off[e], M_per[e], K, s);
        cudaStreamSynchronize(s);
        cudaFree(x);
    }
    const void* b_e(int e) const { return static_cast<const char*>(b) + (e * b_stride); }
    const void* sfb_e(int e) const { return static_cast<const char*>(b_sf) + (e * sfb_stride); }
    void release() {
        for (auto& e : experts)
            free_expert(e);
        for (void* p : {a, a_sf, b, b_sf})
            cudaFree(p);
    }
};

// Host-args form; out filled with NaN first.
static bool run_host_args(const Problem& p, void* out, cudaStream_t s) {
    cudaMemset(out, 0xff, static_cast<size_t>(p.M_total) * p.N * sizeof(half));
    std::vector<size_t> sfa(p.ne);
    std::vector<const void*> hb(p.ne), hsfb(p.ne);
    for (int e = 0; e < p.ne; ++e) {
        sfa[e] = static_cast<size_t>(p.sfa_off[e]);
        hb[e] = p.b_e(e);
        hsfb[e] = p.sfb_e(e);
    }
    GroupedNvfp4HostArgs args{};
    args.offsets = p.offsets.data();
    args.sfa_offsets = sfa.data();
    args.B = hb.data();
    args.SFB = hsfb.data();
    args.alpha = p.alpha.data();
    args.base_A_packed = p.a;
    args.base_A_sf = p.a_sf;
    args.base_D = out;
    return gemm_grouped_nvfp4_host_args(p.ne, p.N, p.K, p.M_total, args, s) &&
           cudaStreamSynchronize(s) == cudaSuccess;
}

static std::vector<half> download(const void* d, size_t n) {
    std::vector<half> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost);
    return h;
}

static size_t bit_mismatches(const std::vector<half>& x, const std::vector<half>& y) {
    size_t n = 0;
    for (size_t i = 0; i < x.size(); ++i)
        n += __half_as_ushort(x[i]) != __half_as_ushort(y[i]) ? 1 : 0;
    return n;
}

// Different kernels (accumulation order), so f16-class tolerance: |d| <= 1e-2 (1 + |ref|) everywhere.
TEST_F(GroupedNvfp4ArgsTest, HostArgsMatchesPerExpertDense) {
    if (sm_ < 120)
        GTEST_SKIP() << "SM120 required";
    if (!cutlass_sm120_nvfp4_available())
        GTEST_SKIP() << "CUTLASS NVFP4 (dense reference) disabled";
    Problem p;
    p.build({32, 0, 16, 48, 64}, /*N=*/256, /*K=*/256, /*seed=*/100, stream_);
    const size_t n_out = static_cast<size_t>(p.M_total) * p.N;

    void* d_ref = nullptr;
    void* d_grp = nullptr;
    cudaMalloc(&d_ref, n_out * sizeof(half));
    cudaMalloc(&d_grp, n_out * sizeof(half));
    for (int e = 0; e < p.ne; ++e) {
        if (p.M_per[e] == 0)
            continue;
        ASSERT_TRUE(
            gemm_nvfp4_cutlass_sm120(static_cast<const char*>(p.a) +
                                         (static_cast<size_t>(p.offsets[e]) * p.K / 2),
                                     static_cast<const char*>(p.a_sf) + p.sfa_off[e], p.experts[e].cutlass_w,
                                     static_cast<half*>(d_ref) + (static_cast<size_t>(p.offsets[e]) * p.N),
                                     p.M_per[e], p.N, p.K, nullptr, 0, stream_))
            << "per-expert dense reference failed on expert " << e;
    }
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    ASSERT_TRUE(run_host_args(p, d_grp, stream_)) << "host-args dispatch failed";
    const std::vector<half> ref = download(d_ref, n_out);
    const std::vector<half> grp = download(d_grp, n_out);

    size_t mismatches = 0;
    double max_abs_err = 0.0;
    double max_ref = 0.0;
    for (size_t i = 0; i < n_out; ++i) {
        const float a = __half2float(ref[i]);
        const float b = __half2float(grp[i]);
        const float err = std::fabs(a - b);
        if (!(err <= 1e-2f * (1.0f + std::fabs(a))))  // NaN counts
            ++mismatches;
        max_abs_err = std::max<double>(max_abs_err, err);
        max_ref = std::max<double>(max_ref, std::fabs(a));
    }
    std::printf("[grouped-args] host vs dense: %zu / %zu mismatches, max_abs_err %.3e, max|ref| %.3g\n",
                mismatches, n_out, max_abs_err, max_ref);
    EXPECT_GT(max_ref, 0.0);
    EXPECT_EQ(mismatches, 0u) << "max_err=" << max_abs_err;

    cudaFree(d_ref);
    cudaFree(d_grp);
    p.release();
}

// Same kernel, only the metadata staging differs: bit-identical. K = 192 ends on a half stage.
class GroupedNvfp4ArgsK : public GroupedNvfp4ArgsTest, public ::testing::WithParamInterface<int> {};

TEST_P(GroupedNvfp4ArgsK, DeviceArgsMatchesHostArgs) {
    if (sm_ < 120)
        GTEST_SKIP() << "SM120 required";
    Problem p;
    p.build({32, 16, 0, 48, 64, 7}, /*N=*/256, GetParam(), /*seed=*/200, stream_);
    const int ne = p.ne;
    const size_t n_out = static_cast<size_t>(p.M_total) * p.N;
    const size_t out_bytes = n_out * sizeof(half);

    void* d_host = nullptr;
    void* d_dev = nullptr;
    cudaMalloc(&d_host, out_bytes);
    cudaMalloc(&d_dev, out_bytes);
    ASSERT_TRUE(run_host_args(p, d_host, stream_)) << "host-args dispatch failed";
    const std::vector<half> ref = download(d_host, n_out);

    int32_t* d_M_per = nullptr;
    int32_t* d_offsets = nullptr;
    int64_t* d_sfa_offsets = nullptr;
    float* d_alpha = nullptr;
    const void** d_B_ptrs = nullptr;
    const void** d_SFB_ptrs = nullptr;
    cudaMalloc(&d_M_per, ne * sizeof(int32_t));
    cudaMalloc(&d_offsets, (ne + 1) * sizeof(int32_t));
    cudaMalloc(&d_sfa_offsets, (ne + 1) * sizeof(int64_t));
    cudaMalloc(&d_alpha, ne * sizeof(float));
    cudaMalloc(&d_B_ptrs, ne * sizeof(const void*));
    cudaMalloc(&d_SFB_ptrs, ne * sizeof(const void*));
    std::vector<const void*> h_B_ptrs(ne), h_SFB_ptrs(ne);
    for (int e = 0; e < ne; ++e) {
        h_B_ptrs[e] = p.b_e(e);
        h_SFB_ptrs[e] = p.sfb_e(e);
    }
    cudaMemcpy(d_M_per, p.M_per.data(), ne * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, p.offsets.data(), (ne + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sfa_offsets, p.sfa_off.data(), (ne + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, p.alpha.data(), ne * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_ptrs, h_B_ptrs.data(), ne * sizeof(const void*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SFB_ptrs, h_SFB_ptrs.data(), ne * sizeof(const void*), cudaMemcpyHostToDevice);

    // Mode (a): B / SFB slab + per-expert byte stride.
    GroupedNvfp4DeviceArgs dargs{};
    dargs.d_M_per = d_M_per;
    dargs.d_expert_offsets = d_offsets;
    dargs.d_sfa_offsets = d_sfa_offsets;
    dargs.d_alpha = d_alpha;
    dargs.base_A_packed = p.a;
    dargs.base_A_sf = p.a_sf;
    dargs.base_B_packed = p.b;
    dargs.b_expert_stride_packed = static_cast<int64_t>(p.b_stride);
    dargs.base_B_sf = p.b_sf;
    dargs.b_expert_stride_sf = static_cast<int64_t>(p.sfb_stride);
    dargs.d_B_ptrs = nullptr;
    dargs.d_SFB_ptrs = nullptr;
    dargs.base_D = d_dev;
    cudaMemset(d_dev, 0xff, out_bytes);
    ASSERT_TRUE(gemm_grouped_nvfp4_tile(ne, p.N, p.K, p.M_total, dargs, stream_)) << "device-args (mode a)";
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    const size_t mism_a = bit_mismatches(ref, download(d_dev, n_out));

    // Mode (b): pointer arrays into the same slab (base + stride ignored).
    GroupedNvfp4DeviceArgs dargs_b = dargs;
    dargs_b.base_B_packed = nullptr;
    dargs_b.b_expert_stride_packed = 0;
    dargs_b.base_B_sf = nullptr;
    dargs_b.b_expert_stride_sf = 0;
    dargs_b.d_B_ptrs = d_B_ptrs;
    dargs_b.d_SFB_ptrs = d_SFB_ptrs;
    cudaMemset(d_dev, 0xff, out_bytes);
    ASSERT_TRUE(gemm_grouped_nvfp4_tile(ne, p.N, p.K, p.M_total, dargs_b, stream_)) << "device-args (mode b)";
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    const size_t mism_b = bit_mismatches(ref, download(d_dev, n_out));

    // Mode (b) in expert blocks of 2: every per-expert array sliced at e0, A / D bases unchanged (absolute
    // prefix sums). The chunked host-expert prefill relies on it.
    cudaMemset(d_dev, 0xff, out_bytes);
    for (int e0 = 0; e0 < ne; e0 += 2) {
        GroupedNvfp4DeviceArgs dargs_c = dargs_b;
        dargs_c.d_M_per = d_M_per + e0;
        dargs_c.d_expert_offsets = d_offsets + e0;
        dargs_c.d_sfa_offsets = d_sfa_offsets + e0;
        dargs_c.d_alpha = d_alpha + e0;
        dargs_c.d_B_ptrs = d_B_ptrs + e0;
        dargs_c.d_SFB_ptrs = d_SFB_ptrs + e0;
        ASSERT_TRUE(gemm_grouped_nvfp4_tile(2, p.N, p.K, p.M_total, dargs_c, stream_)) << "block e0=" << e0;
    }
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    const size_t mism_c = bit_mismatches(ref, download(d_dev, n_out));

    double max_ref = 0.0;
    for (const half& v : ref)
        max_ref = std::max(max_ref, std::fabs(static_cast<double>(__half2float(v))));
    std::printf(
        "[grouped-args] K=%d device vs host bit mismatches: mode a %zu, mode b %zu, blocks %zu / %zu, "
        "max|ref| %.3g\n",
        p.K, mism_a, mism_b, mism_c, n_out, max_ref);
    EXPECT_TRUE(std::isfinite(max_ref) && max_ref > 0.0) << "host-args output empty or non-finite";
    EXPECT_EQ(mism_a, 0u) << "device-args (mode a) differs from host-args";
    EXPECT_EQ(mism_b, 0u) << "device-args (mode b) differs from host-args";
    EXPECT_EQ(mism_c, 0u) << "device-args in expert blocks differs from host-args";

    cudaFree(d_host);
    cudaFree(d_dev);
    cudaFree(d_M_per);
    cudaFree(d_offsets);
    cudaFree(d_sfa_offsets);
    cudaFree(d_alpha);
    cudaFree(d_B_ptrs);
    cudaFree(d_SFB_ptrs);
    p.release();
}

INSTANTIATE_TEST_SUITE_P(K, GroupedNvfp4ArgsK, ::testing::Values(256, 192));

}  // namespace
}  // namespace imp
