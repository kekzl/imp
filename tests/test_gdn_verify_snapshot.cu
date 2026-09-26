// The single-sequence default GDN scans (fused f32/bf16, chunkwise f32) write the speculative-verify
// snapshot: the state after row d_snap_n. They dropped h_snap, so a fully rejected n-gram verify on
// Qwen3.8 adopted an unwritten slab (tool calls cut mid-call in 5 of 16 fresh processes).
#include <gtest/gtest.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/gdn.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

namespace imp {
namespace {

constexpr int kHeads = 8, kHeadDim = 128, kStateSize = 128, kGroups = 8, kRows = 17;

void fill(std::vector<float>& v, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& x : v)
        x = d(rng);
}

class GdnVerifySnapshotTest : public ::testing::Test {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

// The snapshot of a 17-row chunk at row 1 must equal a 1-row scan from the same state. The slab
// starts as NaN, so an unwritten one fails on the finite check.
template <typename StateT>
void check_snapshot_is_first_row_state(int variant) {
    const int conv_channels = 2 * kGroups * kStateSize + kHeads * kHeadDim;
    const size_t h_elems = static_cast<size_t>(kHeads) * kStateSize * kHeadDim;
    std::vector<float> h_conv(static_cast<size_t>(kRows) * conv_channels), af(kRows * kHeads), bf(kRows * kHeads),
        h_Alog(kHeads), h_dtb(kHeads), h0f(h_elems);
    fill(h_conv, 11);
    fill(af, 12, -2.0f, 2.0f);
    fill(bf, 13, -2.0f, 2.0f);
    fill(h_Alog, 14, -4.0f, -0.5f);
    fill(h_dtb, 15);
    fill(h0f, 16, -0.5f, 0.5f);
    std::vector<half> ha(af.size()), hb(bf.size());
    for (size_t i = 0; i < af.size(); ++i) {
        ha[i] = __float2half(af[i]);
        hb[i] = __float2half(bf[i]);
    }
    std::vector<StateT> h0(h_elems);
    for (size_t i = 0; i < h_elems; ++i)
        h0[i] = static_cast<StateT>(h0f[i]);

    float *d_conv, *d_Alog, *d_dtb;
    half *d_a, *d_b, *d_y;
    StateT *d_chunk_h, *d_one_h, *d_snap;
    int* d_snap_n;
    ASSERT_EQ(cudaMalloc(&d_conv, h_conv.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_a, ha.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, hb.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_Alog, h_Alog.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dtb, h_dtb.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(kRows) * kHeads * kHeadDim * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_chunk_h, h_elems * sizeof(StateT)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_one_h, h_elems * sizeof(StateT)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_snap, h_elems * sizeof(StateT)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_snap_n, sizeof(int)), cudaSuccess);
    const int one = 1;
    cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, ha.data(), ha.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hb.data(), hb.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Alog, h_Alog.data(), h_Alog.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dtb, h_dtb.data(), h_dtb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunk_h, h0.data(), h_elems * sizeof(StateT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_one_h, h0.data(), h_elems * sizeof(StateT), cudaMemcpyHostToDevice);
    cudaMemset(d_snap, 0xFF, h_elems * sizeof(StateT));
    cudaMemcpy(d_snap_n, &one, sizeof(int), cudaMemcpyHostToDevice);

    auto scan = [&](StateT* h, int n, StateT* snap, const int* snap_n) {
        if constexpr (std::is_same_v<StateT, float>) {
            if (variant == 0)
                gdn_scan_fused_f32(d_conv, conv_channels, d_a, d_b, d_Alog, d_dtb, h, d_y, n, kHeads, kHeadDim,
                                   kStateSize, kGroups, 0, 0, nullptr, snap, snap_n);
            else
                gdn_scan_chunkwise_f32(d_conv, conv_channels, d_a, d_b, d_Alog, d_dtb, h, d_y, n, kHeads, kHeadDim,
                                       kStateSize, kGroups, 0, 64, 0, nullptr, snap, snap_n);
        } else {
            gdn_scan_fused_bf16(d_conv, conv_channels, d_a, d_b, d_Alog, d_dtb, h, d_y, n, kHeads, kHeadDim,
                                kStateSize, kGroups, 0, 0, nullptr, snap, snap_n);
        }
    };
    scan(d_chunk_h, kRows, d_snap, d_snap_n);
    scan(d_one_h, 1, nullptr, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<StateT> snap(h_elems), one_row(h_elems);
    cudaMemcpy(snap.data(), d_snap, h_elems * sizeof(StateT), cudaMemcpyDeviceToHost);
    cudaMemcpy(one_row.data(), d_one_h, h_elems * sizeof(StateT), cudaMemcpyDeviceToHost);
    double max_diff = 0.0;
    size_t non_finite = 0;
    for (size_t i = 0; i < h_elems; ++i) {
        const double a = static_cast<float>(snap[i]), b = static_cast<float>(one_row[i]);
        if (!std::isfinite(a))
            ++non_finite;
        else
            max_diff = std::max(max_diff, std::fabs(a - b));
    }
    EXPECT_EQ(non_finite, 0u) << "snapshot slab left unwritten";
    EXPECT_LE(max_diff, 1e-6) << "snapshot is not the state after row 1";
    for (void* p : {static_cast<void*>(d_conv), static_cast<void*>(d_a), static_cast<void*>(d_b),
                    static_cast<void*>(d_Alog), static_cast<void*>(d_dtb), static_cast<void*>(d_y),
                    static_cast<void*>(d_chunk_h), static_cast<void*>(d_one_h), static_cast<void*>(d_snap),
                    static_cast<void*>(d_snap_n)})
        cudaFree(p);
}

TEST_F(GdnVerifySnapshotTest, FusedF32WritesTheVerifySnapshot) { check_snapshot_is_first_row_state<float>(0); }
TEST_F(GdnVerifySnapshotTest, ChunkwiseF32WritesTheVerifySnapshot) { check_snapshot_is_first_row_state<float>(1); }
TEST_F(GdnVerifySnapshotTest, FusedBf16WritesTheVerifySnapshot) {
    check_snapshot_is_first_row_state<__nv_bfloat16>(0);
}

}  // namespace
}  // namespace imp
