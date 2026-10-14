// Phase 4c (gemm.q8_drop_source): the Q8_0 IMMA planes are the weight's only 8-bit copy.
// - planes -> FP16 is bit-equal to dequant_gpu on the Q8_0 source;
// - gguf_source_fallback (IMMA decline) on a weight without source computes from the planes:
//   the source is poisoned and the output still matches dequant_gpu + gemm bit for bit.

#include "compute/gemm.h"
#include "compute/mmq_q8_imma.h"
#include "core/weight_handle.h"
#include "exec/gemm_dispatch_fallback.h"
#include "exec/quant_scratch.h"
#include "exec/weight_caches.h"
#include "quant/dequant_gpu.h"
#include "scoped_engine_arena.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace imp {

IMP_TEST_ENGINE_ARENA(64ull << 20);

namespace {

constexpr int kN = 256;
constexpr int kK = 512;
constexpr int kBlockBytes = 34;  // half d + 32 s8
constexpr size_t kElems = static_cast<size_t>(kN) * kK;

std::vector<uint8_t> random_q8(unsigned seed) {
    std::vector<uint8_t> w(static_cast<size_t>(kN) * (kK / 32) * kBlockBytes);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> dd(0.001f, 0.05f);
    for (size_t b = 0; b < w.size() / kBlockBytes; ++b) {
        const __half d = __float2half(dd(rng));
        std::memcpy(w.data() + b * kBlockBytes, &d, 2);
        for (int i = 0; i < 32; ++i)
            w[b * kBlockBytes + 2 + i] = static_cast<uint8_t>(static_cast<int8_t>(qd(rng)));
    }
    return w;
}

std::vector<uint16_t> download(const void* d, size_t n) {
    std::vector<uint16_t> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(uint16_t), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

// A device Q8_0 weight, its dequant_gpu FP16 reference and its IMMA planes.
struct Q8Weight {
    void* src = nullptr;
    __half* ref16 = nullptr;
    explicit Q8Weight(unsigned seed) {
        const auto h = random_q8(seed);
        EXPECT_EQ(cudaMalloc(&src, h.size()), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(src, h.data(), h.size(), cudaMemcpyHostToDevice), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&ref16, kElems * sizeof(__half)), cudaSuccess);
        dequant_gpu(src, ref16, QType::Q8_0, kN, kK, nullptr);
        EXPECT_TRUE(mmq_q8_imma_build_planes(src, kN, kK, nullptr));
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    ~Q8Weight() {
        mmq_q8_imma_release_all();
        cudaFree(src);
        cudaFree(ref16);
    }
};

TEST(Q8PlanesOnly, PlanesToF16MatchesDequantGpu) {
    Q8Weight w(11);
    EXPECT_EQ(mmq_q8_imma_plane_bytes_used(), imma_q8_plane_bytes(kN, kK));
    __half* out = nullptr;
    ASSERT_EQ(cudaMalloc(&out, kElems * sizeof(__half)), cudaSuccess);
    ASSERT_TRUE(mmq_q8_imma_planes_to_f16(w.src, out, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(download(out, kElems), download(w.ref16, kElems));
    mmq_q8_imma_release_all();
    EXPECT_FALSE(mmq_q8_imma_planes_to_f16(w.src, out, nullptr)) << "release_all kept the planes";
    cudaFree(out);
}

TEST(Q8PlanesOnly, BuildDeclinesShapesTheKernelDeclines) {
    void* src = nullptr;
    ASSERT_EQ(cudaMalloc(&src, 4096), cudaSuccess);
    EXPECT_FALSE(mmq_q8_imma_build_planes(src, 3, 64, nullptr));  // N odd
    EXPECT_FALSE(mmq_q8_imma_build_planes(src, 2, 96, nullptr));  // K % 64
    cudaFree(src);
}

// beta 0 and 1 (the o / down residual form): planes route vs dequant_gpu + gemm, source poisoned.
TEST(Q8PlanesOnly, DeclineFallbackComputesFromThePlanes) {
    gemm_init();
    constexpr int kM = 5;
    Q8Weight w(13);
    ASSERT_EQ(cudaMemset(w.src, 0x7f, static_cast<size_t>(kN) * (kK / 32) * kBlockBytes), cudaSuccess);

    std::vector<__half> hx(static_cast<size_t>(kM) * kK), hy0(static_cast<size_t>(kM) * kN);
    std::mt19937 rng(14);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    for (auto& v : hx)
        v = __float2half(xd(rng));
    for (auto& v : hy0)
        v = __float2half(xd(rng));
    __half *x = nullptr, *y = nullptr, *yref = nullptr, *scratch = nullptr;
    const size_t ybytes = hy0.size() * sizeof(__half);
    ASSERT_EQ(cudaMalloc(&x, hx.size() * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&y, ybytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&yref, ybytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&scratch, kElems * sizeof(__half)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(x, hx.data(), hx.size() * sizeof(__half), cudaMemcpyHostToDevice), cudaSuccess);

    WeightCaches wc;
    wc.q8_planes_only.insert(w.src);
    QuantScratch qs;
    qs.dequant = scratch;
    qs.dequant_size = kElems * sizeof(__half);
    WeightHandle h;
    h.source_data = w.src;
    h.source_qtype = QType::Q8_0;
    h.shape[0] = kN;
    h.shape[1] = kK;
    const int64_t xs[2] = {kM, kK}, ys[2] = {kM, kN}, ws[2] = {kN, kK};
    const Tensor tx(x, QType::F16, 2, xs, true);
    Tensor ty(y, QType::F16, 2, ys, true), tyref(yref, QType::F16, 2, ys, true);
    const Tensor tw(w.ref16, QType::F16, 2, ws, true);
    for (const float beta : {0.0f, 1.0f}) {
        ASSERT_EQ(cudaMemcpy(y, hy0.data(), ybytes, cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(yref, hy0.data(), ybytes, cudaMemcpyHostToDevice), cudaSuccess);
        GemmContext ctx;
        ctx.beta = beta;
        ctx.wcache = &wc;
        ctx.qscratch = &qs;
        gguf_source_fallback(h, tx, ty, ctx, /*decode_rows=*/false);
        gemm(tx, tw, tyref, 1.0f, beta, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(download(y, hy0.size()), download(yref, hy0.size())) << "beta " << beta;
    }
    cudaFree(x);
    cudaFree(y);
    cudaFree(yref);
    cudaFree(scratch);
}

}  // namespace
}  // namespace imp
