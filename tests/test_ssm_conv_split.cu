#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/ssm.h"
#include "core/tensor.h"

#include <cstring>
#include <random>
#include <vector>

#include "test_cuda_skip.h"

namespace imp {
namespace {

// Fused conv1d + SiLU + split (ssm_conv_split.cu) vs ssm_conv1d_prefill + silu_inplace +
// host split: bitwise equal x/B/C planes and conv window, input rows at the projection pitch.
struct SplitCase {
    int n_tokens, real_n;  // real_n < 0: no device length
    bool with_state, with_bias;
};

struct SplitOut {
    std::vector<uint16_t> x, b, c;
    std::vector<float> state;
};

constexpr int kInner = 512, kBc = 128, kK = 4, kChannels = kInner + (2 * kBc), kPitch = kChannels + 520;

template <typename T>
T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

SplitOut run_split(const SplitCase& c, bool fused, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    std::vector<half> proj(static_cast<size_t>(c.n_tokens) * kPitch), w(kChannels * kK), bias(kChannels);
    for (auto& e : proj)
        e = __float2half(u(rng));
    for (auto& e : w)
        e = __float2half(0.5f * u(rng));
    for (auto& e : bias)
        e = __float2half(0.5f * u(rng));
    std::vector<float> st(kChannels * kK);
    for (auto& e : st)
        e = u(rng);
    const int off = 64;  // xBC column offset inside a projection row
    half* d_proj = upload(proj);
    half* d_w = upload(w);
    half* d_bias = upload(bias);
    float* d_state = c.with_state ? upload(st) : nullptr;
    int* d_n = upload(std::vector<int>{c.real_n});
    const int64_t w_shape[2] = {kChannels, kK}, b_shape[1] = {kChannels};
    Tensor tw(d_w, QType::F16, 2, w_shape, true);
    Tensor tb = c.with_bias ? Tensor(d_bias, QType::F16, 1, b_shape, true) : Tensor();
    const int n = c.n_tokens;
    const int* real = c.real_n >= 0 ? d_n : nullptr;

    SplitOut o;
    o.x.resize(static_cast<size_t>(n) * kInner);
    o.b.resize(static_cast<size_t>(n) * kBc);
    o.c.resize(static_cast<size_t>(n) * kBc);
    if (fused) {
        half *x, *bc;
        cudaMalloc(&x, o.x.size() * 2);
        cudaMalloc(&bc, (o.b.size() + o.c.size()) * 2);
        EXPECT_TRUE(ssm_conv1d_prefill_silu_split(d_state, d_proj + off, kPitch, tw, tb, x, bc,
                                                  bc + (static_cast<size_t>(n) * kBc), n, kInner, kBc, kK,
                                                  nullptr));
        ssm_conv1d_commit(d_state, d_proj + off, kPitch, n, kChannels, kK, real, nullptr);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaMemcpy(o.x.data(), x, o.x.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(o.b.data(), bc, o.b.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(o.c.data(), bc + o.b.size(), o.c.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(x);
        cudaFree(bc);
    } else {
        half *in, *out;
        cudaMalloc(&in, static_cast<size_t>(n) * kChannels * 2);
        cudaMalloc(&out, static_cast<size_t>(n) * kChannels * 2);
        cudaMemcpy2D(in, kChannels * 2, d_proj + off, kPitch * 2, kChannels * 2, n, cudaMemcpyDeviceToDevice);
        const int64_t shape[2] = {n, kChannels};
        Tensor tin(in, QType::F16, 2, shape, true), tout(out, QType::F16, 2, shape, true);
        ssm_conv1d_prefill(d_state, tin, tw, tb, tout, kK, nullptr, real);
        silu_inplace(tout, nullptr);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        std::vector<uint16_t> all(static_cast<size_t>(n) * kChannels);
        cudaMemcpy(all.data(), out, all.size() * 2, cudaMemcpyDeviceToHost);
        for (int t = 0; t < n; ++t) {
            std::memcpy(&o.x[static_cast<size_t>(t) * kInner], &all[static_cast<size_t>(t) * kChannels],
                        kInner * 2);
            std::memcpy(&o.b[static_cast<size_t>(t) * kBc],
                        &all[(static_cast<size_t>(t) * kChannels) + kInner], kBc * 2);
            std::memcpy(&o.c[static_cast<size_t>(t) * kBc],
                        &all[(static_cast<size_t>(t) * kChannels) + kInner + kBc], kBc * 2);
        }
        cudaFree(in);
        cudaFree(out);
    }
    if (d_state) {
        o.state.resize(st.size());
        cudaMemcpy(o.state.data(), d_state, st.size() * 4, cudaMemcpyDeviceToHost);
    }
    for (void* p : {static_cast<void*>(d_proj), static_cast<void*>(d_w), static_cast<void*>(d_bias),
                    static_cast<void*>(d_state), static_cast<void*>(d_n)})
        cudaFree(p);
    return o;
}

TEST(SSMConvSplitTest, BitIdenticalToSeparateKernels) {
    SKIP_IF_NO_CUDA();
    const SplitCase cases[] = {{1, -1, true, true},    {3, -1, true, false},   {64, -1, false, true},
                               {300, 250, true, true}, {2048, -1, true, true}, {77, 2, true, true}};
    uint32_t seed = 3;
    for (const auto& c : cases) {
        const SplitOut a = run_split(c, false, seed), b = run_split(c, true, seed);
        ++seed;
        EXPECT_EQ(a.x, b.x) << "n=" << c.n_tokens;
        EXPECT_EQ(a.b, b.b) << "n=" << c.n_tokens;
        EXPECT_EQ(a.c, b.c) << "n=" << c.n_tokens;
        EXPECT_EQ(a.state, b.state) << "n=" << c.n_tokens;
    }
}

TEST(SSMConvSplitTest, RefusesUncoveredShapes) {
    SKIP_IF_NO_CUDA();
    const Tensor none;
    alignas(16) static half buf[64];
    EXPECT_FALSE(
        ssm_conv1d_prefill_silu_split(nullptr, buf, 16, none, none, buf, buf, buf, 4, 8, 8, 3, nullptr));
    EXPECT_FALSE(
        ssm_conv1d_prefill_silu_split(nullptr, buf + 1, 16, none, none, buf, buf, buf, 4, 8, 8, 4, nullptr));
}

}  // namespace
}  // namespace imp
