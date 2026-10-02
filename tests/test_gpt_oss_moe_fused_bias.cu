// gpt-oss MoE prefill bias fusion (#2466): the fused act+quantize (gate/up bias + clamped GLU)
// and the fused scatter (down bias) must be byte-identical to the 4-launch sequence they
// replace. Golden hashes pin the non-gpt-oss fused paths to the pre-#2466 bytes.

#include <gtest/gtest.h>
#include "compute/activation.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/moe_routing.h"
#include "core/tensor.h"
#include "model/model_config.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace imp {
namespace {

// Deterministic generator independent of the standard library's distributions.
struct Lcg {
    uint64_t s;
    float next(float lo, float hi) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return lo + (hi - lo) * static_cast<float>(s >> 40) / static_cast<float>(1ULL << 24);
    }
};

uint64_t fnv1a(const std::vector<uint8_t>& b) {
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t c : b)
        h = (h ^ c) * 1099511628211ULL;
    return h;
}

struct DevBuf {
    void* p = nullptr;
    size_t n = 0;
    explicit DevBuf(size_t bytes) : n(bytes) {
        EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
        EXPECT_EQ(cudaMemset(p, 0, bytes), cudaSuccess);
    }
    ~DevBuf() { cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    template <class T>
    void upload(const std::vector<T>& h) {
        ASSERT_EQ(h.size() * sizeof(T), n);
        ASSERT_EQ(cudaMemcpy(p, h.data(), n, cudaMemcpyHostToDevice), cudaSuccess);
    }
    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> h(n);
        EXPECT_EQ(cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost), cudaSuccess);
        return h;
    }
    template <class T>
    T* as() const {
        return static_cast<T*>(p);
    }
};

std::vector<half> to_half(const std::vector<float>& f) {
    std::vector<half> h(f.size());
    for (size_t i = 0; i < f.size(); ++i)
        h[i] = __float2half(f[i]);
    return h;
}

// gpt-oss-20b expert width; rows per expert cover an empty expert, a 1-row expert and a
// 128-row SfAtom tile crossing.
constexpr int kK = 2880;
const std::vector<int> kRowsPerExpert = {5, 0, 130, 1, 37, 0, 64, 9};

struct MoeQuantFixture {
    int ne = static_cast<int>(kRowsPerExpert.size());
    int expanded = 0;
    std::vector<int> offsets;
    size_t sf_bytes = 0;  // per expert
    std::unique_ptr<DevBuf> d_offsets, d_sfa_ptrs, d_sfa, d_packed;

    MoeQuantFixture() {
        offsets.push_back(0);
        for (int r : kRowsPerExpert)
            offsets.push_back(offsets.back() + r);
        expanded = offsets.back();
        sf_bytes = cutlass_nvfp4_sf_size(*std::max_element(kRowsPerExpert.begin(), kRowsPerExpert.end()), kK);
        d_offsets = std::make_unique<DevBuf>(offsets.size() * sizeof(int));
        d_offsets->upload(offsets);
        d_sfa = std::make_unique<DevBuf>(sf_bytes * ne);
        std::vector<uint8_t*> ptrs(ne);
        for (int e = 0; e < ne; ++e)
            ptrs[e] = d_sfa->as<uint8_t>() + sf_bytes * e;
        d_sfa_ptrs = std::make_unique<DevBuf>(ptrs.size() * sizeof(uint8_t*));
        d_sfa_ptrs->upload(ptrs);
        d_packed = std::make_unique<DevBuf>(static_cast<size_t>(expanded) * kK / 2);
    }
    void clear() {
        ASSERT_EQ(cudaMemset(d_sfa->p, 0, d_sfa->n), cudaSuccess);
        ASSERT_EQ(cudaMemset(d_packed->p, 0, d_packed->n), cudaSuccess);
    }
    uint8_t* const* sfa_bases() const { return d_sfa_ptrs->as<uint8_t* const>(); }
    const int* offs() const { return d_offsets->as<const int>(); }
};

TEST(GptOssMoeFusedBias, FusedGluQuantizeIsByteIdenticalToTheFourLaunchSequence) {
    MoeQuantFixture fx;
    const size_t n = static_cast<size_t>(fx.expanded) * kK;
    Lcg rng{0x2466};
    std::vector<float> gate(n), up(n), gb(static_cast<size_t>(fx.ne) * kK), ub(gb.size());
    for (auto& v : gate)
        v = rng.next(-12.0f, 12.0f);
    for (auto& v : up)
        v = rng.next(-12.0f, 12.0f);
    for (auto& v : gb)
        v = rng.next(-2.0f, 2.0f);
    for (auto& v : ub)
        v = rng.next(-2.0f, 2.0f);
    // Clamp edges: pre-bias values at and around +-7, bias pushing across the limit both ways.
    const float edges[] = {7.0f, -7.0f, 6.99f, 7.01f, -6.99f, -7.01f, 0.0f, 65.0f, -65.0f};
    for (size_t i = 0; i < n; i += 97) {
        gate[i] = edges[(i / 97) % 9];
        up[i] = edges[(i / 97 + 4) % 9];
    }
    for (size_t i = 0; i < gb.size(); i += 13) {
        gb[i] = (i & 1) ? 0.5f : -0.5f;
        ub[i] = (i & 2) ? 0.5f : 0.0f;
    }
    // Host check that the clamp is actually exercised after the FP16 bias add.
    int above = 0, up_clamped = 0;
    for (int e = 0; e < fx.ne; ++e)
        for (int r = fx.offsets[e]; r < fx.offsets[e + 1]; ++r)
            for (int k = 0; k < kK; ++k) {
                const size_t i = static_cast<size_t>(r) * kK + k, b = static_cast<size_t>(e) * kK + k;
                above += __half2float(__float2half(__half2float(__float2half(gate[i])) +
                                                   __half2float(__float2half(gb[b])))) >= 7.0f;
                up_clamped += std::fabs(up[i] + ub[b]) >= 7.0f;
            }
    ASSERT_GT(above, 1000);
    ASSERT_GT(up_clamped, 1000);

    DevBuf d_gate(n * 2), d_up(n * 2), d_gate_ref(n * 2), d_up_ref(n * 2), d_act(n * 2);
    DevBuf d_gb(gb.size() * 2), d_ub(ub.size() * 2);
    d_gate.upload(to_half(gate));
    d_up.upload(to_half(up));
    d_gate_ref.upload(to_half(gate));
    d_up_ref.upload(to_half(up));
    d_gb.upload(to_half(gb));
    d_ub.upload(to_half(ub));

    // Reference: bias x2, gpt_oss_glu into an FP16 buffer, plain MoE quantize (pre-#2466 path).
    fx.clear();
    moe_add_expert_bias_sorted(d_gate_ref.p, d_gb.p, fx.offs(), fx.ne, fx.expanded, kK, nullptr);
    moe_add_expert_bias_sorted(d_up_ref.p, d_ub.p, fx.offs(), fx.ne, fx.expanded, kK, nullptr);
    int64_t shape[2] = {fx.expanded, kK};
    Tensor tg(d_gate_ref.p, QType::F16, 2, shape, true), tu(d_up_ref.p, QType::F16, 2, shape, true),
        ta(d_act.p, QType::F16, 2, shape, true);
    gpt_oss_glu(tg, tu, ta, nullptr);
    quantize_fp16_to_nvfp4_cutlass_moe(d_act.p, fx.d_packed->p, fx.sfa_bases(), fx.offs(), fx.expanded, kK,
                                       fx.ne, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto ref_packed = fx.d_packed->bytes();
    const auto ref_sfa = fx.d_sfa->bytes();
    size_t nonzero = 0;
    for (uint8_t b : ref_packed)
        nonzero += b != 0;
    ASSERT_GT(nonzero, ref_packed.size() / 2) << "reference is degenerate";

    fx.clear();
    fused_act_quantize_fp16_to_nvfp4_cutlass_moe(d_gate.p, d_up.p, fx.d_packed->p, fx.sfa_bases(), fx.offs(),
                                                 fx.expanded, kK, fx.ne, FFNActivation::GPT_OSS_GLU, nullptr,
                                                 d_gb.p, d_ub.p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto got_packed = fx.d_packed->bytes();
    const auto got_sfa = fx.d_sfa->bytes();
    size_t diff_packed = 0, diff_sfa = 0;
    for (size_t i = 0; i < got_packed.size(); ++i)
        diff_packed += got_packed[i] != ref_packed[i];
    for (size_t i = 0; i < got_sfa.size(); ++i)
        diff_sfa += got_sfa[i] != ref_sfa[i];
    EXPECT_EQ(diff_packed, 0u) << "of " << got_packed.size() << " packed bytes";
    EXPECT_EQ(diff_sfa, 0u) << "of " << got_sfa.size() << " SFA bytes";
    // Inputs are read-only in the fused kernel (the old path mutated gate/up in place).
    EXPECT_EQ(fnv1a(d_gate.bytes()), fnv1a([&] {
                  std::vector<uint8_t> b(n * 2);
                  auto h = to_half(gate);
                  std::memcpy(b.data(), h.data(), b.size());
                  return b;
              }()));
}

// Pre-#2466 bytes of the SWIGLU / GEGLU / RELU_SQR fused quantize (captured on main 49236dfb).
TEST(GptOssMoeFusedBias, NonGptOssFusedQuantizeBytesUnchanged) {
    struct Case {
        FFNActivation act;
        uint64_t packed, sfa;
    };
    const Case cases[] = {
        {FFNActivation::SWIGLU, 0x1c24c6fe86cfb925ULL, 0x7fe27dd1648c5ea1ULL},
        {FFNActivation::GEGLU, 0xd7f6e169e5392491ULL, 0x970b4e8a5e58c280ULL},
        {FFNActivation::RELU_SQR, 0x52e2b4f45adfdbb0ULL, 0x65adf94defc54b89ULL},
    };
    MoeQuantFixture fx;
    const size_t n = static_cast<size_t>(fx.expanded) * kK;
    Lcg rng{0x604};
    std::vector<float> gate(n), up(n);
    for (auto& v : gate)
        v = rng.next(-6.0f, 6.0f);
    for (auto& v : up)
        v = rng.next(-6.0f, 6.0f);
    DevBuf d_gate(n * 2), d_up(n * 2);
    d_gate.upload(to_half(gate));
    d_up.upload(to_half(up));
    for (const auto& c : cases) {
        fx.clear();
        fused_act_quantize_fp16_to_nvfp4_cutlass_moe(c.act == FFNActivation::RELU_SQR ? nullptr : d_gate.p,
                                                     d_up.p, fx.d_packed->p, fx.sfa_bases(), fx.offs(),
                                                     fx.expanded, kK, fx.ne, c.act, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const uint64_t hp = fnv1a(fx.d_packed->bytes()), hs = fnv1a(fx.d_sfa->bytes());
        EXPECT_EQ(hp, c.packed) << "act " << static_cast<int>(c.act);
        EXPECT_EQ(hs, c.sfa) << "act " << static_cast<int>(c.act);
    }
}

// Routing on the host: token t picks experts (5t + 3k) % ne, rows stable-sorted by expert.
struct ScatterFixture {
    int n = 37, top_k = 4, ne = 8, d = kK;
    int expanded = n * top_k;
    std::vector<int32_t> expert_idx, t2e, offsets;
    std::vector<float> weights;
    ScatterFixture() {
        expert_idx.resize(expanded);
        weights.resize(expanded);
        Lcg rng{0x547};
        for (int t = 0; t < n; ++t)
            for (int k = 0; k < top_k; ++k) {
                expert_idx[t * top_k + k] = (t * 5 + 3 * k) % ne;
                weights[t * top_k + k] = rng.next(0.05f, 0.6f);
            }
        offsets.assign(ne + 1, 0);
        for (int e : expert_idx)
            offsets[e + 1]++;
        for (int e = 0; e < ne; ++e)
            offsets[e + 1] += offsets[e];
        t2e.resize(expanded);
        std::vector<int32_t> fill(offsets.begin(), offsets.end() - 1);
        for (int f = 0; f < expanded; ++f)
            t2e[f] = fill[expert_idx[f]]++;
    }
};

TEST(GptOssMoeFusedBias, ScatterDownBiasIsByteIdenticalToSortedBiasThenScatter) {
    ScatterFixture sf;
    Lcg rng{0x2453};
    std::vector<float> down(static_cast<size_t>(sf.expanded) * sf.d), bias(static_cast<size_t>(sf.ne) * sf.d),
        res(static_cast<size_t>(sf.n) * sf.d);
    for (auto& v : down)
        v = rng.next(-40.0f, 40.0f);
    for (auto& v : bias)
        v = rng.next(-3.0f, 3.0f);
    for (auto& v : res)
        v = rng.next(-8.0f, 8.0f);
    DevBuf d_down(down.size() * 2), d_down_ref(down.size() * 2), d_bias(bias.size() * 2),
        d_res(res.size() * 2), d_out_ref(res.size() * 2), d_out(res.size() * 2);
    DevBuf d_idx(sf.expanded * 4), d_t2e(sf.expanded * 4), d_w(sf.expanded * 4),
        d_offs(sf.offsets.size() * 4);
    d_down.upload(to_half(down));
    d_down_ref.upload(to_half(down));
    d_bias.upload(to_half(bias));
    d_res.upload(to_half(res));
    d_idx.upload(sf.expert_idx);
    d_t2e.upload(sf.t2e);
    d_w.upload(sf.weights);
    d_offs.upload(sf.offsets);

    for (const bool with_residual : {true, false}) {
        const void* r = with_residual ? d_res.p : nullptr;
        ASSERT_EQ(cudaMemcpy(d_down_ref.p, d_down.p, d_down.n, cudaMemcpyDeviceToDevice), cudaSuccess);
        moe_add_expert_bias_sorted(d_down_ref.p, d_bias.p, d_offs.as<const int32_t>(), sf.ne, sf.expanded,
                                   sf.d, nullptr);
        moe_scatter_fused_residual(d_down_ref.p, d_t2e.as<const int32_t>(), d_w.as<const float>(), r,
                                   d_out_ref.p, sf.n, sf.d, sf.top_k, nullptr);
        moe_scatter_fused_residual(d_down.p, d_t2e.as<const int32_t>(), d_w.as<const float>(), r, d_out.p,
                                   sf.n, sf.d, sf.top_k, nullptr, d_bias.p, d_idx.as<const int32_t>());
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto a = d_out_ref.bytes(), b = d_out.bytes();
        size_t diff = 0;
        for (size_t i = 0; i < a.size(); ++i)
            diff += a[i] != b[i];
        EXPECT_EQ(diff, 0u) << "residual=" << with_residual << " of " << a.size() << " bytes";
        // The bias must matter, or the comparison proves nothing.
        moe_scatter_fused_residual(d_down.p, d_t2e.as<const int32_t>(), d_w.as<const float>(), r, d_out.p,
                                   sf.n, sf.d, sf.top_k, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto c = d_out.bytes();
        size_t moved = 0;
        for (size_t i = 0; i < a.size(); ++i)
            moved += a[i] != c[i];
        EXPECT_GT(moved, a.size() / 4);
    }
}

// Pre-#2466 bytes of the bias-free fused scatter (captured on main 49236dfb).
TEST(GptOssMoeFusedBias, ScatterWithoutBiasBytesUnchanged) {
    ScatterFixture sf;
    Lcg rng{0x2488};
    std::vector<float> down(static_cast<size_t>(sf.expanded) * sf.d), res(static_cast<size_t>(sf.n) * sf.d);
    for (auto& v : down)
        v = rng.next(-40.0f, 40.0f);
    for (auto& v : res)
        v = rng.next(-8.0f, 8.0f);
    DevBuf d_down(down.size() * 2), d_res(res.size() * 2), d_out(res.size() * 2);
    DevBuf d_t2e(sf.expanded * 4), d_w(sf.expanded * 4);
    d_down.upload(to_half(down));
    d_res.upload(to_half(res));
    d_t2e.upload(sf.t2e);
    d_w.upload(sf.weights);
    moe_scatter_fused_residual(d_down.p, d_t2e.as<const int32_t>(), d_w.as<const float>(), d_res.p, d_out.p,
                               sf.n, sf.d, sf.top_k, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const uint64_t h = fnv1a(d_out.bytes());
    EXPECT_EQ(h, 0xcb30e1291595d33bULL);
}

}  // namespace
}  // namespace imp
