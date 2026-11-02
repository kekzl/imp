// fp32_residual_norm (compute/fp32_residual_norm.h) against the unfused launches it replaces in the
// Gemma-4 decode layer: bit-equal FP32 residual, FP16 views, normed output and NVFP4 activations,
// rows 1-32, Gemma-3/4 and Qwen3 hidden sizes, both norm weight offsets.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/fp32_residual_norm.h"
#include "compute/layernorm.h"
#include "exec/executor_kernels.h"
#include "exec/executor_forward_moe_kernels.cuh"
#include "quant/nvfp4_quant.h"
#include "core/tensor.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace imp {

// Defined in executor_forward.cu: the per-layer output scalar of Gemma-4.
__global__ void scale_fp16_by_fp16ptr_kernel(half* __restrict__ data, const half* __restrict__ scale_ptr,
                                             int64_t n);
__global__ void scale_fp32_by_fp16ptr_kernel(float* __restrict__ data, const half* __restrict__ scale_ptr,
                                             int64_t n);

namespace {

struct Buf {
    void* p = nullptr;
    size_t bytes = 0;
    explicit Buf(size_t b) : bytes(b) { cudaMalloc(&p, b); }
    ~Buf() { cudaFree(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    template <typename T>
    T* as() const {
        return static_cast<T*>(p);
    }
    std::vector<uint8_t> host() const {
        std::vector<uint8_t> v(bytes);
        cudaMemcpy(v.data(), p, bytes, cudaMemcpyDeviceToHost);
        return v;
    }
};

Tensor view(const Buf& b, QType q, int rows, int cols) {
    int64_t shape[2] = {rows, cols};
    return Tensor(b.p, q, 2, shape, true);
}
Tensor view1(const Buf& b, int cols) {
    int64_t shape[1] = {cols};
    return Tensor(b.p, QType::F16, 1, shape, true);
}

void upload_half(Buf& b, const std::vector<float>& v) {
    std::vector<half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        h[i] = __float2half(v[i]);
    cudaMemcpy(b.p, h.data(), h.size() * sizeof(half), cudaMemcpyHostToDevice);
}

// Deterministic pseudo-random values in [-amp, amp].
std::vector<float> fill(size_t n, uint32_t seed, float amp) {
    std::vector<float> v(n);
    uint32_t s = seed * 2654435761u + 12345u;
    for (auto& x : v) {
        s = (s * 1664525u) + 1013904223u;
        x = amp * ((static_cast<float>(s >> 8) / 8388608.0f) - 1.0f);
    }
    return v;
}

struct Case {
    int rows;
    int d;
    float w_off;
};

// rows x d inputs; row 2 of the branch inputs carries +-Inf / NaN (sanitize), row 3 of the FP32
// residual sits above 65000 (the accumulate kernel's FP16 overflow scaling).
struct Inputs {
    int rows, d;
    Buf a, b, fp32, wa, wb, wacc, wn, scale;
    Inputs(int r, int dm)
        : rows(r),
          d(dm),
          a(sizeof(half) * r * dm),
          b(sizeof(half) * r * dm),
          fp32(sizeof(float) * r * dm),
          wa(sizeof(half) * dm),
          wb(sizeof(half) * dm),
          wacc(sizeof(half) * dm),
          wn(sizeof(half) * dm),
          scale(sizeof(half)) {
        auto va = fill(static_cast<size_t>(r) * dm, 1, 40.0f);
        auto vb = fill(static_cast<size_t>(r) * dm, 2, 900.0f);
        if (r > 2) {
            va[(2 * static_cast<size_t>(dm)) + 5] = INFINITY;
            vb[(2 * static_cast<size_t>(dm)) + 7] = -INFINITY;
            vb[(2 * static_cast<size_t>(dm)) + 300] = NAN;
        }
        upload_half(a, va);
        upload_half(b, vb);
        auto vf = fill(static_cast<size_t>(r) * dm, 3, 30.0f);
        if (r > 3)
            for (int c = 0; c < dm; c += 97)
                vf[(3 * static_cast<size_t>(dm)) + c] = 70000.0f + static_cast<float>(c);
        cudaMemcpy(fp32.p, vf.data(), fp32.bytes, cudaMemcpyHostToDevice);
        upload_half(wa, fill(dm, 4, 2.0f));
        upload_half(wb, fill(dm, 5, 2.0f));
        upload_half(wacc, fill(dm, 6, 2.0f));
        upload_half(wn, fill(dm, 7, 2.0f));
        upload_half(scale, {0.71875f});
    }
};

void expect_same(const Buf& x, const Buf& y, const char* what, const Case& c) {
    EXPECT_EQ(0, memcmp(x.host().data(), y.host().data(), x.bytes))
        << what << " differs, rows=" << c.rows << " d=" << c.d << " w_off=" << c.w_off;
}

const Case kCases[] = {
    {1, 2816, 0.0f}, {2, 2816, 0.0f},  {5, 2816, 1.0f},  {16, 2816, 0.0f}, {32, 2816, 1.0f},
    {7, 2048, 0.0f}, {16, 4096, 0.0f}, {32, 2560, 1.0f}, {16, 3840, 1.0f},
};
constexpr float kEps = 1e-6f;

TEST(Fp32ResidualNorm, NormQuantMatchesNormThenQuantize) {
    for (const Case& c : kCases) {
        Inputs in(c.rows, c.d);
        const size_t hb = sizeof(half) * c.rows * c.d;
        Buf no_ref(hb), no_f(hb), pk_ref(c.rows * c.d / 2), pk_f(c.rows * c.d / 2), sc_ref(c.rows * c.d / 16),
            sc_f(c.rows * c.d / 16);
        Tensor no_t = view(no_ref, QType::F16, c.rows, c.d);
        rmsnorm_fp32_to_fp16(view(in.fp32, QType::F32, c.rows, c.d), view1(in.wn, c.d), no_t, kEps, nullptr,
                             c.w_off);
        quantize_fp16_to_nvfp4_into(no_ref.p, c.rows, c.d, pk_ref.as<uint8_t>(), sc_ref.as<uint8_t>(), 1.0f,
                                    nullptr);

        Fp32ResidArgs a;
        a.d_model = c.d;
        a.eps = kEps;
        a.w_off = c.w_off;
        a.fp32 = in.fp32.as<float>();
        a.w_norm = in.wn.as<half>();
        a.no_out = no_f.as<half>();
        a.xq_packed = pk_f.as<uint8_t>();
        a.xq_scales = sc_f.as<uint8_t>();
        ASSERT_TRUE(fp32_residual_norm(Fp32ResidMode::kNorm, a, c.rows, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        expect_same(no_ref, no_f, "no", c);
        expect_same(pk_ref, pk_f, "packed", c);
        expect_same(sc_ref, sc_f, "scales", c);
    }
}

TEST(Fp32ResidualNorm, PostAttnMatchesAccumCopyNorm) {
    for (const Case& c : kCases) {
        Inputs ref(c.rows, c.d), fus(c.rows, c.d);
        const size_t hb = sizeof(half) * c.rows * c.d;
        Buf h_ref(hb), r_ref(hb), no_ref(hb), h_f(hb), r_f(hb), no_f(hb);
        rmsnorm_fp32_accum_to_fp16(ref.a.as<half>(), ref.wacc.as<half>(), ref.fp32.as<float>(),
                                   h_ref.as<half>(), c.rows, c.d, kEps, c.w_off, nullptr);
        cudaMemcpy(r_ref.p, h_ref.p, hb, cudaMemcpyDeviceToDevice);
        Tensor no_t = view(no_ref, QType::F16, c.rows, c.d);
        rmsnorm_fp32_to_fp16(view(ref.fp32, QType::F32, c.rows, c.d), view1(ref.wn, c.d), no_t, kEps, nullptr,
                             c.w_off);

        Fp32ResidArgs a;
        a.d_model = c.d;
        a.eps = kEps;
        a.w_off = c.w_off;
        a.in_a = fus.a.as<half>();
        a.w_accum = fus.wacc.as<half>();
        a.fp32 = fus.fp32.as<float>();
        a.h_out = h_f.as<half>();
        a.r_out = r_f.as<half>();
        a.w_norm = fus.wn.as<half>();
        a.no_out = no_f.as<half>();
        ASSERT_TRUE(fp32_residual_norm(Fp32ResidMode::kPostAttn, a, c.rows, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        expect_same(ref.fp32, fus.fp32, "fp32 residual", c);
        expect_same(h_ref, h_f, "h", c);
        expect_same(r_ref, r_f, "r", c);
        expect_same(no_ref, no_f, "no", c);
    }
}

// Gemma-4 layer tail: sanitize + post_ffw_norm_2 (MoE branch), sanitize + post_ffw_norm_1 (dense
// branch), add, post_ffn_norm into the FP32 residual, layer_out_scale, next layer's attn_norm + quantize.
void run_tail_case(const Case& c, bool next_norm) {
    Inputs ref(c.rows, c.d), fus(c.rows, c.d);
    const int64_t total = static_cast<int64_t>(c.rows) * c.d;
    const size_t hb = sizeof(half) * total;
    Buf no_ref(hb), no_f(hb), pk_ref(total / 2), pk_f(total / 2), sc_ref(total / 16), sc_f(total / 16);
    Tensor ta = view(ref.a, QType::F16, c.rows, c.d);
    Tensor tb = view(ref.b, QType::F16, c.rows, c.d);
    sanitize_fp16(ref.a.as<half>(), total, nullptr);
    rmsnorm(ta, view1(ref.wa, c.d), ta, kEps, nullptr, c.w_off);
    sanitize_fp16(ref.b.as<half>(), total, nullptr);
    rmsnorm(tb, view1(ref.wb, c.d), tb, kEps, nullptr, c.w_off);
    elementwise_add(ta, tb, nullptr);
    rmsnorm_fp32_accum_to_fp16(ref.a.as<half>(), ref.wacc.as<half>(), ref.fp32.as<float>(), ref.a.as<half>(),
                               c.rows, c.d, kEps, c.w_off, nullptr);
    scale_fp16_by_fp16ptr_kernel<<<static_cast<int>((total / 2 + 255) / 256), 256>>>(ref.a.as<half>(),
                                                                                     ref.scale.as<half>(),
                                                                                     total);
    scale_fp32_by_fp16ptr_kernel<<<static_cast<int>((total + 255) / 256), 256>>>(ref.fp32.as<float>(),
                                                                                 ref.scale.as<half>(), total);
    if (next_norm) {
        Tensor no_t = view(no_ref, QType::F16, c.rows, c.d);
        rmsnorm_fp32_to_fp16(view(ref.fp32, QType::F32, c.rows, c.d), view1(ref.wn, c.d), no_t, kEps, nullptr,
                             c.w_off);
        quantize_fp16_to_nvfp4_into(no_ref.p, c.rows, c.d, pk_ref.as<uint8_t>(), sc_ref.as<uint8_t>(), 1.0f,
                                    nullptr);
    }

    Fp32ResidArgs a;
    a.d_model = c.d;
    a.eps = kEps;
    a.w_off = c.w_off;
    a.in_a = fus.a.as<half>();
    a.w_a = fus.wa.as<half>();
    a.in_b = fus.b.as<half>();
    a.w_b = fus.wb.as<half>();
    a.w_accum = fus.wacc.as<half>();
    a.fp32 = fus.fp32.as<float>();
    a.h_out = fus.a.as<half>();  // in place, as phase 8 runs it
    a.layer_scale = fus.scale.as<half>();
    if (next_norm) {
        a.w_norm = fus.wn.as<half>();
        a.no_out = no_f.as<half>();
        a.xq_packed = pk_f.as<uint8_t>();
        a.xq_scales = sc_f.as<uint8_t>();
    }
    ASSERT_TRUE(fp32_residual_norm(Fp32ResidMode::kTail, a, c.rows, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expect_same(ref.fp32, fus.fp32, "fp32 residual", c);
    expect_same(ref.a, fus.a, "h", c);
    if (next_norm) {
        expect_same(no_ref, no_f, "no", c);
        expect_same(pk_ref, pk_f, "packed", c);
        expect_same(sc_ref, sc_f, "scales", c);
    }
}

TEST(Fp32ResidualNorm, TailMatchesUnfusedChain) {
    for (const Case& c : kCases) {
        run_tail_case(c, /*next_norm=*/true);
        run_tail_case(c, /*next_norm=*/false);
    }
}

// A row's bits do not depend on how many rows share the launch (#2152).
TEST(Fp32ResidualNorm, RowBitsIndependentOfM) {
    const int d = 2816;
    Inputs big(32, d), one(1, d);
    // Row 0 of `big` equals `one` (same seeds, same row-major prefix).
    const size_t hb = sizeof(half) * d;
    Buf no_big(hb * 32), no_one(hb), pk_big(32 * d / 2), pk_one(d / 2), sc_big(32 * d / 16), sc_one(d / 16);
    auto args = [&](Inputs& in, Buf& no, Buf& pk, Buf& sc) {
        Fp32ResidArgs a;
        a.d_model = d;
        a.eps = kEps;
        a.in_a = in.a.as<half>();
        a.w_a = in.wa.as<half>();
        a.in_b = in.b.as<half>();
        a.w_b = in.wb.as<half>();
        a.w_accum = in.wacc.as<half>();
        a.fp32 = in.fp32.as<float>();
        a.h_out = in.a.as<half>();
        a.layer_scale = in.scale.as<half>();
        a.w_norm = in.wn.as<half>();
        a.no_out = no.as<half>();
        a.xq_packed = pk.as<uint8_t>();
        a.xq_scales = sc.as<uint8_t>();
        return a;
    };
    ASSERT_TRUE(fp32_residual_norm(Fp32ResidMode::kTail, args(big, no_big, pk_big, sc_big), 32, nullptr));
    ASSERT_TRUE(fp32_residual_norm(Fp32ResidMode::kTail, args(one, no_one, pk_one, sc_one), 1, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(0, memcmp(no_big.host().data(), no_one.host().data(), hb));
    EXPECT_EQ(0, memcmp(pk_big.host().data(), pk_one.host().data(), d / 2));
    EXPECT_EQ(0, memcmp(big.fp32.host().data(), one.fp32.host().data(), sizeof(float) * d));
}

TEST(Fp32ResidualNorm, RefusesOutsideEnvelope) {
    Inputs in(2, 2816);
    Buf no(sizeof(half) * 2 * 2816);
    Fp32ResidArgs a;
    a.eps = kEps;
    a.fp32 = in.fp32.as<float>();
    a.w_norm = in.wn.as<half>();
    a.no_out = no.as<half>();
    a.d_model = 2816 + 128;  // not a multiple of 256
    EXPECT_FALSE(fp32_residual_norm(Fp32ResidMode::kNorm, a, 2, nullptr));
    a.d_model = 5376;  // Gemma-3 27B: above the accumulate kernel's two-chunk mapping
    EXPECT_FALSE(fp32_residual_norm(Fp32ResidMode::kNorm, a, 2, nullptr));
    a.d_model = 1536;  // kTail below rmsnorm()'s row-block range
    a.in_a = a.in_b = in.a.as<half>();
    a.w_a = a.w_b = a.w_accum = in.wa.as<half>();
    a.h_out = in.a.as<half>();
    EXPECT_FALSE(fp32_residual_norm(Fp32ResidMode::kTail, a, 2, nullptr));
}

}  // namespace
}  // namespace imp
