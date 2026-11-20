// The single-sequence default GDN scans (fused f32/bf16, chunkwise f32) write the speculative-verify
// snapshot: the state after row d_snap_n. They dropped h_snap, so a fully rejected n-gram verify on
// Qwen3.8 adopted an unwritten slab (tool calls cut mid-call in 5 of 16 fresh processes).
#include <gtest/gtest.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/gdn.h"
#include "compute/ssm.h"
#include "core/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <type_traits>
#include <vector>

namespace imp {
namespace {

constexpr int kHeads = 8, kHeadDim = 128, kStateSize = 128, kGroups = 8, kRows = 17;

enum Variant { kFusedF32, kChunkwiseF32 };

void fill(std::vector<float>& v, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& x : v)
        x = d(rng);
}

Tensor make_fp16_gpu(const float* host, std::initializer_list<int64_t> shape) {
    Tensor t;
    t.qtype = QType::F16;
    t.ndim = static_cast<int>(shape.size());
    int i = 0;
    for (auto s : shape)
        t.shape[i++] = s;
    t.compute_strides();
    t.on_device = true;
    cudaMalloc(&t.data, t.nbytes());
    std::vector<half> h(t.numel());
    for (int64_t j = 0; j < t.numel(); j++)
        h[j] = __float2half(host[j]);
    cudaMemcpy(t.data, h.data(), t.nbytes(), cudaMemcpyHostToDevice);
    return t;
}

class GdnVerifySnapshotTest : public ::testing::Test {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

// The snapshot of a `rows`-row scan at row `snap_row` must equal a `snap_row`-row fused scan from
// the same state. The slab starts as NaN, so an unwritten one fails on the finite check.
template <typename StateT>
void check_snapshot_is_row_state(int variant, int rows = kRows, int snap_row = 1, bool second = false) {
    // second: the slab at snap2_off must hold the state after snap_row + 1 rows, the first slab still
    // the state after snap_row.
    const int ref_rows = snap_row + (second ? 1 : 0);
    const int conv_channels = 2 * kGroups * kStateSize + kHeads * kHeadDim;
    const size_t h_elems = static_cast<size_t>(kHeads) * kStateSize * kHeadDim;
    std::vector<float> h_conv(static_cast<size_t>(rows) * conv_channels), af(rows * kHeads),
        bf(rows * kHeads), h_Alog(kHeads), h_dtb(kHeads), h0f(h_elems);
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
    ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(rows) * kHeads * kHeadDim * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_chunk_h, h_elems * sizeof(StateT)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_one_h, h_elems * sizeof(StateT)), cudaSuccess);
    const int64_t snap2_off = second ? static_cast<int64_t>(h_elems * sizeof(StateT)) : 0;
    ASSERT_EQ(cudaMalloc(&d_snap, 2 * h_elems * sizeof(StateT)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_snap_n, sizeof(int)), cudaSuccess);
    cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, ha.data(), ha.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hb.data(), hb.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Alog, h_Alog.data(), h_Alog.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dtb, h_dtb.data(), h_dtb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chunk_h, h0.data(), h_elems * sizeof(StateT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_one_h, h0.data(), h_elems * sizeof(StateT), cudaMemcpyHostToDevice);
    cudaMemset(d_snap, 0xFF, 2 * h_elems * sizeof(StateT));
    cudaMemcpy(d_snap_n, &snap_row, sizeof(int), cudaMemcpyHostToDevice);

    half* const y16 = reinterpret_cast<half*>(d_y);
    auto scan = [&](int v, StateT* h, int n, StateT* snap, const int* snap_n, int64_t off = 0) {
        if constexpr (std::is_same_v<StateT, float>) {
            if (v == kFusedF32)
                gdn_scan_fused_f32(d_conv, conv_channels, d_a, d_b, d_Alog, d_dtb, h, y16, n, kHeads,
                                   kHeadDim, kStateSize, kGroups, 0, 0, nullptr, snap, snap_n, off);
            else
                gdn_scan_chunkwise_f32(d_conv, conv_channels, d_a, d_b, d_Alog, d_dtb, h, y16, n, kHeads,
                                       kHeadDim, kStateSize, kGroups, 0, 64, 0, nullptr, snap, snap_n, off);
        } else {
            gdn_scan_fused_bf16(d_conv, conv_channels, d_a, d_b, d_Alog, d_dtb, h, y16, n, kHeads, kHeadDim,
                                kStateSize, kGroups, 0, 0, nullptr, snap, snap_n, off);
        }
    };
    scan(variant, d_chunk_h, rows, d_snap, d_snap_n, snap2_off);
    scan(kFusedF32, d_one_h, ref_rows, nullptr, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<StateT> snap(h_elems), one_row(h_elems);
    cudaMemcpy(snap.data(), d_snap + (second ? h_elems : 0), h_elems * sizeof(StateT),
               cudaMemcpyDeviceToHost);
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
    EXPECT_LE(max_diff, 1e-6) << "snapshot is not the state after row " << ref_rows;
    std::printf("snapshot variant=%d rows=%d snap_row=%d second=%d max_abs_diff=%.3g\n", variant, rows,
                snap_row, static_cast<int>(second), max_diff);
    if (second) {
        // The first slab still holds the state after snap_row rows.
        cudaMemcpy(d_one_h, h0.data(), h_elems * sizeof(StateT), cudaMemcpyHostToDevice);
        scan(kFusedF32, d_one_h, snap_row, nullptr, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        cudaMemcpy(snap.data(), d_snap, h_elems * sizeof(StateT), cudaMemcpyDeviceToHost);
        cudaMemcpy(one_row.data(), d_one_h, h_elems * sizeof(StateT), cudaMemcpyDeviceToHost);
        double first_diff = 0.0, slabs_diff = 0.0;
        for (size_t i = 0; i < h_elems; ++i)
            first_diff = std::max(first_diff, std::fabs(static_cast<double>(static_cast<float>(snap[i])) -
                                                        static_cast<double>(static_cast<float>(one_row[i]))));
        EXPECT_LE(first_diff, 1e-6) << "first slab is not the state after row " << snap_row;
        std::vector<StateT> two(h_elems);
        cudaMemcpy(two.data(), d_snap + h_elems, h_elems * sizeof(StateT), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < h_elems; ++i)
            slabs_diff = std::max(slabs_diff, std::fabs(static_cast<double>(static_cast<float>(snap[i])) -
                                                        static_cast<double>(static_cast<float>(two[i]))));
        EXPECT_GT(slabs_diff, 1e-4) << "both slabs hold the same state: the test proves nothing";
    }
    for (void* p : {static_cast<void*>(d_conv), static_cast<void*>(d_a), static_cast<void*>(d_b),
                    static_cast<void*>(d_Alog), static_cast<void*>(d_dtb), static_cast<void*>(d_y),
                    static_cast<void*>(d_chunk_h), static_cast<void*>(d_one_h), static_cast<void*>(d_snap),
                    static_cast<void*>(d_snap_n)})
        cudaFree(p);
}

TEST_F(GdnVerifySnapshotTest, FusedF32WritesTheVerifySnapshot) {
    check_snapshot_is_row_state<float>(kFusedF32);
}
TEST_F(GdnVerifySnapshotTest, ChunkwiseF32WritesTheVerifySnapshot) {
    check_snapshot_is_row_state<float>(kChunkwiseF32);
}
TEST_F(GdnVerifySnapshotTest, FusedBf16WritesTheVerifySnapshot) {
    check_snapshot_is_row_state<__nv_bfloat16>(kFusedF32);
}
// #2214: 65 rows > chunk 64 threw; 64 rows took the chunkwise kernel, which has no snapshot write.
TEST_F(GdnVerifySnapshotTest, ChunkwiseF32SnapshotPastOneChunk) {
    check_snapshot_is_row_state<float>(kChunkwiseF32, 65, 1);
    check_snapshot_is_row_state<float>(kChunkwiseF32, 65, 64);
}
// 3-row MTP verify chunk, one of two drafts accepted: the state after row 2 comes from the second slab.
TEST_F(GdnVerifySnapshotTest, SecondSlabIsTheStateOneRowLater) {
    check_snapshot_is_row_state<float>(kFusedF32, 3, 1, /*second=*/true);
    check_snapshot_is_row_state<__nv_bfloat16>(kFusedF32, 3, 1, /*second=*/true);
    check_snapshot_is_row_state<float>(kChunkwiseF32, 17, 1, /*second=*/true);
}
TEST_F(GdnVerifySnapshotTest, ChunkwiseF32SnapshotOnAFullChunk) {
    check_snapshot_is_row_state<float>(kChunkwiseF32, 64, 33);
}

// GDN conv, 3-row verify chunk: the first slab holds the window after 1 row, the slab at snap2_off the
// window after 2 rows (one of two drafts accepted); both equal a plain 1- and 2-row conv's commit.
TEST_F(GdnVerifySnapshotTest, ConvSecondSlabIsTheWindowOneRowLater) {
    constexpr int channels = 40, K = 4, n_chunk = 3;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> h_x(static_cast<size_t>(n_chunk) * channels), h_w(static_cast<size_t>(channels) * K),
        h_b(channels), h_state0(static_cast<size_t>(channels) * K);
    for (auto* v : {&h_x, &h_w, &h_b, &h_state0})
        for (auto& e : *v)
            e = dist(rng);
    const size_t st = h_state0.size();
    Tensor d_w = make_fp16_gpu(h_w.data(), {channels, K});
    Tensor d_b = make_fp16_gpu(h_b.data(), {channels});
    float *d_st = nullptr, *d_prev = nullptr, *d_snap = nullptr, *d_out = nullptr;
    int* d_snap_n = nullptr;
    ASSERT_EQ(cudaMalloc(&d_st, st * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_prev, st * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_snap, 2 * st * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, static_cast<size_t>(n_chunk) * channels * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_snap_n, sizeof(int)), cudaSuccess);
    const int one = 1;
    cudaMemcpy(d_snap_n, &one, sizeof(int), cudaMemcpyHostToDevice);

    // Reference: the committed window of a plain n-row conv from the same state.
    auto commit_after = [&](int n) {
        Tensor d_x = make_fp16_gpu(h_x.data(), {n, channels});
        cudaMemcpy(d_st, h_state0.data(), st * sizeof(float), cudaMemcpyHostToDevice);
        ssm_conv1d_prefill_f32_silu(d_st, d_x, d_w, d_b, d_out, K, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> out(st);
        cudaMemcpy(out.data(), d_st, st * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_x.data);
        return out;
    };
    const std::vector<float> ref1 = commit_after(1), ref2 = commit_after(2);

    Tensor d_x = make_fp16_gpu(h_x.data(), {n_chunk, channels});
    cudaMemcpy(d_st, h_state0.data(), st * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev, h_state0.data(), st * sizeof(float), cudaMemcpyHostToDevice);
    std::vector<float> poison(2 * st, 12345.0f);
    cudaMemcpy(d_snap, poison.data(), 2 * st * sizeof(float), cudaMemcpyHostToDevice);
    ssm_conv1d_prefill_f32_silu(d_st, d_x, d_w, d_b, d_out, K, nullptr, nullptr, d_snap, d_snap_n, d_prev,
                                static_cast<int64_t>(st * sizeof(float)));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> snaps(2 * st);
    cudaMemcpy(snaps.data(), d_snap, 2 * st * sizeof(float), cudaMemcpyDeviceToHost);
    bool differs = false;
    for (size_t i = 0; i < st; i++) {
        EXPECT_NEAR(snaps[i], ref1[i], 1e-5f) << "first slab, i=" << i;
        EXPECT_NEAR(snaps[st + i], ref2[i], 1e-5f) << "second slab, i=" << i;
        differs = differs || std::fabs(ref1[i] - ref2[i]) > 1e-4f;
    }
    EXPECT_TRUE(differs) << "1-row and 2-row windows are identical: the test proves nothing";
    cudaFree(d_x.data);
    cudaFree(d_w.data);
    cudaFree(d_b.data);
    for (void* p : {static_cast<void*>(d_st), static_cast<void*>(d_prev), static_cast<void*>(d_snap),
                    static_cast<void*>(d_out), static_cast<void*>(d_snap_n)})
        cudaFree(p);
}

}  // namespace
}  // namespace imp
