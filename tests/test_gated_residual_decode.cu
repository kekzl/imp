// hc_read_decode / hc_write / hc_mix(out2) against the unfused Qwen4Exp gated-residual kernels:
// bit-identical outputs at n = 1 (Flash-Next shape hc=4, d=2560, lowrank=320) and refusal paths.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/gated_residual.h"
#include "compute/gemm.h"
#include "core/tensor.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace imp;

namespace {

struct DevBuf {
    void* p = nullptr;
    explicit DevBuf(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DevBuf() {
        if (p)
            cudaFree(p);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

std::vector<__half> rand_half(size_t n, float scale, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-scale, scale);
    std::vector<__half> v(n);
    for (auto& e : v)
        e = __float2half(u(rng));
    return v;
}

void upload(DevBuf& b, const std::vector<__half>& v) {
    cudaMemcpy(b.p, v.data(), v.size() * sizeof(__half), cudaMemcpyHostToDevice);
}

std::vector<uint16_t> download(const DevBuf& b, size_t n) {
    std::vector<uint16_t> v(n);
    cudaMemcpy(v.data(), b.p, n * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    return v;
}

Tensor t2(const DevBuf& b, int64_t r, int64_t c) {
    const int64_t s[2] = {r, c};
    return Tensor(b.p, QType::F16, 2, s, true);
}

void check_read(int hc, int d, int lowrank, bool with_inject) {
    const int K = hc * d;
    const float eps = 1e-6f;
    DevBuf x(K * 2), w(K * 2), down(static_cast<size_t>(lowrank) * K * 2),
        inj_w(static_cast<size_t>(hc) * K * 2);
    DevBuf normed_ref(K * 2), low_ref(lowrank * 2), inj_ref(hc * 2);
    DevBuf normed(K * 2), low(lowrank * 2), inj(hc * 2);
    upload(x, rand_half(K, 2.0f, 1));
    upload(w, rand_half(K, 0.5f, 2));
    upload(down, rand_half(static_cast<size_t>(lowrank) * K, 0.02f, 3));
    upload(inj_w, rand_half(static_cast<size_t>(hc) * K, 0.02f, 4));
    cudaMemset(inj_ref.p, 0, hc * 2);
    cudaMemset(inj.p, 0, hc * 2);

    const int64_t w_shape[1] = {K};
    Tensor tx = t2(x, 1, K), tw(w.p, QType::F16, 1, w_shape, true), tdown = t2(down, lowrank, K),
           tinj_w = t2(inj_w, hc, K);
    Tensor tn_ref = t2(normed_ref, 1, K), tl_ref = t2(low_ref, 1, lowrank), ti_ref = t2(inj_ref, 1, hc);
    Tensor tn = t2(normed, 1, K), tl = t2(low, 1, lowrank), ti = t2(inj, 1, hc);

    hc_grouped_rmsnorm(tx, tw, tn_ref, hc, d, eps, nullptr);
    gemm(tn_ref, tdown, tl_ref, 1.0f, 0.0f, nullptr);
    hc_silu_div(tl_ref, hc, nullptr);
    if (with_inject) {
        gemm(tn_ref, tinj_w, ti_ref, 1.0f, 0.0f, nullptr);
        hc_inject_weights(ti_ref, hc, nullptr);
    }
    ASSERT_TRUE(
        hc_read_decode(tx, tw, tdown, with_inject ? &tinj_w : nullptr, tn, tl, ti, hc, d, eps, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(download(normed, K), download(normed_ref, K));
    EXPECT_EQ(download(low, lowrank), download(low_ref, lowrank));
    EXPECT_EQ(download(inj, hc), download(inj_ref, hc));
}

}  // namespace

TEST(GatedResidualDecode, ReadBitIdenticalFlashNextShape) { check_read(4, 2560, 320, true); }

TEST(GatedResidualDecode, ReadBitIdenticalNoInject) { check_read(4, 2560, 320, false); }

TEST(GatedResidualDecode, ReadBitIdenticalSmallShape) { check_read(2, 1024, 64, true); }

TEST(GatedResidualDecode, ReadRefusesUnsupported) {
    const int hc = 4, d = 2560, K = hc * d, lowrank = 320;
    DevBuf x(2 * K * 2), w(K * 2), down(static_cast<size_t>(lowrank) * K * 2), o(2 * K * 2),
        l(2 * lowrank * 2), i(2 * hc * 2);
    const int64_t w_shape[1] = {K};
    Tensor tw(w.p, QType::F16, 1, w_shape, true), tdown = t2(down, lowrank, K);
    Tensor tx2 = t2(x, 2, K), tn = t2(o, 2, K), tl = t2(l, 2, lowrank), ti = t2(i, 2, hc);
    EXPECT_FALSE(hc_read_decode(tx2, tw, tdown, nullptr, tn, tl, ti, hc, d, 1e-6f, nullptr));  // n = 2
    Tensor tx1 = t2(x, 1, K);
    EXPECT_FALSE(hc_read_decode(tx1, tw, tdown, nullptr, tn, tl, ti, 9, K / 9, 1e-6f, nullptr));  // hc > 8
}

TEST(GatedResidualDecode, WriteAndMixMatchUnfused) {
    const int n = 3, hc = 4, d = 2560, K = hc * d;
    DevBuf hid_ref(n * K * 2), hid(n * K * 2), h(n * d * 2), mixed(n * d * 2), inj(n * hc * 2),
        out(n * d * 2);
    const auto hid0 = rand_half(static_cast<size_t>(n) * K, 1.0f, 5);
    upload(hid_ref, hid0);
    upload(hid, hid0);
    upload(h, rand_half(static_cast<size_t>(n) * d, 1.0f, 6));
    upload(mixed, rand_half(static_cast<size_t>(n) * d, 1.0f, 7));
    upload(inj, rand_half(static_cast<size_t>(n) * hc, 2.0f, 8));
    Tensor thr = t2(hid_ref, n, K), th = t2(hid, n, K), tH = t2(h, n, d), tm = t2(mixed, n, d),
           ti = t2(inj, n, hc), to = t2(out, n, d);
    hc_sub(tH, tm, to, nullptr);
    hc_inject_add(thr, to, ti, hc, d, nullptr);
    hc_write(th, tH, tm, ti, hc, d, nullptr);

    DevBuf mixw(n * K * 2), normed(n * K * 2), m1(n * d * 2), m2(n * d * 2);
    upload(mixw, rand_half(static_cast<size_t>(n) * K, 3.0f, 9));
    upload(normed, rand_half(static_cast<size_t>(n) * K, 1.0f, 10));
    Tensor tmw = t2(mixw, n, K), tno = t2(normed, n, K), tm1 = t2(m1, n, d), tm2 = t2(m2, n, d);
    hc_mix(tmw, tno, tm1, hc, d, nullptr, &tm2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_EQ(download(hid, static_cast<size_t>(n) * K), download(hid_ref, static_cast<size_t>(n) * K));
    EXPECT_EQ(download(m2, static_cast<size_t>(n) * d), download(m1, static_cast<size_t>(n) * d));
}
