// Batched GDN scan (N independent sequences in one launch): concurrent GDN decode used to
// run one sequence per step, forcing the WHOLE decode step (including ordinary FFN/attention
// GEMMs) onto the M=1 path. Profiled under 32-way load: CUTLASS GEMM 1.0% of GPU time on a
// GDN model vs 71.8% dense; M=1 GEMV ~82% vs ~7%; the scan itself only 3.8%.
// Sequences parallelise across blockIdx.y (each owns its own recurrent-state slot, shares
// only weights). Tolerance not bit-compare: SPLIT=2 (two threads/state column, vs SPLIT=1
// single-sequence) sums in a different FP order (measured: 2/1024 FP16 outputs, ~7% of FP32
// state words differ at rounding level). A slot mix-up instead misplaces a whole sequence's
// state, so 1e-3 rel (output) / 1e-4 (state) separates rounding from a wrong slot by orders
// of magnitude.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/gdn_factor.cuh"
#include <cuda_fp16.h>
#include "compute/gdn.h"
#include "compute/ssm.h"

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace imp {
namespace {

struct ScanShape {
    int n_seq;
    int n_tokens;
    int n_heads = 8;
    int head_dim = 128;
    int state_size = 128;
    int n_groups = 8;
};

// Deterministic pseudo-random fill; the values only have to be non-degenerate.
void fill(std::vector<float>& v, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& x : v)
        x = d(rng);
}

class GdnBatchedScanTest : public ::testing::Test {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

// Run `shape` both ways and compare. Returns false if the device refuses the
// allocation (skips rather than fails).
void run_and_compare(const ScanShape& s, const std::vector<int>& slots) {
    const int conv_channels = 2 * s.n_groups * s.state_size + s.n_heads * s.head_dim;
    const int inner = s.n_heads * s.head_dim;
    const size_t rows = static_cast<size_t>(s.n_seq) * s.n_tokens;
    const size_t state_elems = static_cast<size_t>(s.n_heads) * s.state_size * s.head_dim;

    std::vector<float> h_conv(rows * conv_channels);
    std::vector<float> h_alpha_f(rows * s.n_heads), h_beta_f(rows * s.n_heads);
    std::vector<float> h_Alog(s.n_heads), h_dtb(s.n_heads);
    fill(h_conv, 1234);
    fill(h_alpha_f, 5678, -2.0f, 2.0f);
    fill(h_beta_f, 9012, -2.0f, 2.0f);
    fill(h_Alog, 3456, -4.0f, -0.5f);
    fill(h_dtb, 7890, -1.0f, 1.0f);

    std::vector<half> h_alpha(h_alpha_f.size()), h_beta(h_beta_f.size());
    for (size_t i = 0; i < h_alpha_f.size(); i++)
        h_alpha[i] = __float2half(h_alpha_f[i]);
    for (size_t i = 0; i < h_beta_f.size(); i++)
        h_beta[i] = __float2half(h_beta_f[i]);

    // One initial state per sequence, distinct so a slot mix-up is visible.
    const int max_slot = *std::max_element(slots.begin(), slots.end());
    const size_t pool_elems = static_cast<size_t>(max_slot + 1) * state_elems;
    std::vector<float> h_pool_init(pool_elems);
    fill(h_pool_init, 2468, -0.5f, 0.5f);

    float *d_conv = nullptr, *d_Alog = nullptr, *d_dtb = nullptr, *d_pool = nullptr;
    half *d_alpha = nullptr, *d_beta = nullptr, *d_y = nullptr;
    int* d_slots = nullptr;
    ASSERT_EQ(cudaMalloc(&d_conv, h_conv.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_alpha, h_alpha.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_beta, h_beta.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_Alog, h_Alog.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dtb, h_dtb.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pool, pool_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, rows * inner * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_slots, slots.size() * sizeof(int)), cudaSuccess);

    cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha.data(), h_alpha.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Alog, h_Alog.data(), h_Alog.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dtb, h_dtb.data(), h_dtb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slots, slots.data(), slots.size() * sizeof(int), cudaMemcpyHostToDevice);

    // ---- arm A: one sequence at a time, the shipped single-sequence launcher
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, rows * inner * sizeof(half));
    for (int i = 0; i < s.n_seq; i++) {
        gdn_scan_fused_f32(d_conv + static_cast<size_t>(i) * s.n_tokens * conv_channels, conv_channels,
                           d_alpha + static_cast<size_t>(i) * s.n_tokens * s.n_heads,
                           d_beta + static_cast<size_t>(i) * s.n_tokens * s.n_heads, d_Alog, d_dtb,
                           d_pool + static_cast<size_t>(slots[i]) * state_elems,
                           d_y + static_cast<size_t>(i) * s.n_tokens * inner, s.n_tokens, s.n_heads,
                           s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> pool_single(pool_elems);
    std::vector<half> y_single(rows * inner);
    cudaMemcpy(pool_single.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_single.data(), d_y, y_single.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // ---- arm B: all sequences in one launch
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, rows * inner * sizeof(half));
    gdn_scan_fused_f32_batched(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb, d_pool, d_slots,
                               static_cast<int64_t>(state_elems), d_y, s.n_seq, s.n_tokens, s.n_heads,
                               s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> pool_batched(pool_elems);
    std::vector<half> y_batched(rows * inner);
    cudaMemcpy(pool_batched.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_batched.data(), d_y, y_batched.size() * sizeof(half), cudaMemcpyDeviceToHost);

    // Compared via RMS of the tensor, not per-element: state values can be arbitrarily close to
    // zero, where elementwise relative error reports 1e-3 for a 1e-9 absolute difference.
    // Scale-relative answers "did this drift relative to the numbers' size" and still separates
    // rounding from a wrong slot by orders of magnitude.
    auto rms = [](const auto& v, auto conv) {
        double ss = 0.0;
        for (const auto& x : v) {
            const double d = conv(x);
            ss += d * d;
        }
        return std::sqrt(ss / static_cast<double>(v.size()));
    };
    const double state_scale = std::max(1e-9, rms(pool_single, [](float x) { return double(x); }));
    const double y_scale = std::max(1e-9, rms(y_single, [](half x) { return double(__half2float(x)); }));

    double worst_state = 0.0, worst_y = 0.0;
    for (size_t i = 0; i < pool_elems; i++)
        worst_state = std::max(worst_state, std::abs(double(pool_single[i]) - double(pool_batched[i])));
    for (size_t i = 0; i < y_single.size(); i++)
        worst_y = std::max(worst_y,
                           std::abs(double(__half2float(y_single[i])) - double(__half2float(y_batched[i]))));
    worst_state /= state_scale;
    worst_y /= y_scale;

    EXPECT_LT(worst_state, 1e-3) << "recurrent state diverges beyond rounding (worst " << worst_state
                                 << " of RMS, n_seq=" << s.n_seq << ") — a wrong slot reads O(1) off";
    EXPECT_LT(worst_y, 1e-2) << "scan output diverges beyond rounding (worst " << worst_y
                             << " of RMS, n_seq=" << s.n_seq << ")";

    // The states must actually have moved — a kernel that wrote nothing would
    // pass both comparisons above.
    size_t moved = 0;
    for (size_t i = 0; i < pool_elems; i++)
        if (pool_batched[i] != h_pool_init[i])
            moved++;
    EXPECT_GT(moved, pool_elems / 4) << "state barely changed — the scan may not have run";

    cudaFree(d_conv);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_Alog);
    cudaFree(d_dtb);
    cudaFree(d_pool);
    cudaFree(d_y);
    cudaFree(d_slots);
}

// Control: is the SINGLE-sequence reference arm itself reproducible? If this
// fails, every comparison above is measuring the reference, not the batch path.
TEST_F(GdnBatchedScanTest, ReferenceArmIsReproducible) {
    const int n_seq = 32, n_tokens = 1, n_heads = 8, head_dim = 128, state_size = 128, n_groups = 8;
    const int conv_channels = 2 * n_groups * state_size + n_heads * head_dim;
    const int inner = n_heads * head_dim;
    const size_t rows = static_cast<size_t>(n_seq) * n_tokens;
    const size_t state_elems = static_cast<size_t>(n_heads) * state_size * head_dim;
    const size_t pool_elems = static_cast<size_t>(n_seq) * state_elems;

    std::vector<float> h_conv(rows * conv_channels), h_af(rows * n_heads), h_bf(rows * n_heads);
    std::vector<float> h_Alog(n_heads), h_dtb(n_heads), h_pool_init(pool_elems);
    fill(h_conv, 1234); fill(h_af, 5678, -2.f, 2.f); fill(h_bf, 9012, -2.f, 2.f);
    fill(h_Alog, 3456, -4.f, -0.5f); fill(h_dtb, 7890, -1.f, 1.f); fill(h_pool_init, 2468, -0.5f, 0.5f);
    std::vector<half> h_a(h_af.size()), h_b(h_bf.size());
    for (size_t i = 0; i < h_af.size(); i++) h_a[i] = __float2half(h_af[i]);
    for (size_t i = 0; i < h_bf.size(); i++) h_b[i] = __float2half(h_bf[i]);

    float *d_conv, *d_Alog, *d_dtb, *d_pool; half *d_alpha, *d_beta, *d_y;
    ASSERT_EQ(cudaMalloc(&d_conv, h_conv.size()*sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_alpha, h_a.size()*sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_beta, h_b.size()*sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_Alog, n_heads*sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dtb, n_heads*sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pool, pool_elems*sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, rows*inner*sizeof(half)), cudaSuccess);
    cudaMemcpy(d_conv, h_conv.data(), h_conv.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_a.data(), h_a.size()*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_b.data(), h_b.size()*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Alog, h_Alog.data(), n_heads*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dtb, h_dtb.data(), n_heads*sizeof(float), cudaMemcpyHostToDevice);

    std::vector<std::vector<float>> pools(2);
    for (int rep = 0; rep < 2; rep++) {
        cudaMemcpy(d_pool, h_pool_init.data(), pool_elems*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, rows*inner*sizeof(half));
        for (int i = 0; i < n_seq; i++)
            gdn_scan_fused_f32(d_conv + static_cast<size_t>(i)*n_tokens*conv_channels, conv_channels,
                               d_alpha + static_cast<size_t>(i)*n_tokens*n_heads,
                               d_beta + static_cast<size_t>(i)*n_tokens*n_heads, d_Alog, d_dtb,
                               d_pool + static_cast<size_t>(i)*state_elems,
                               d_y + static_cast<size_t>(i)*n_tokens*inner, n_tokens, n_heads,
                               head_dim, state_size, n_groups, nullptr, 1);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        pools[rep].resize(pool_elems);
        cudaMemcpy(pools[rep].data(), d_pool, pool_elems*sizeof(float), cudaMemcpyDeviceToHost);
    }
    size_t d = 0;
    for (size_t i = 0; i < pool_elems; i++) if (pools[0][i] != pools[1][i]) d++;
    EXPECT_EQ(d, 0u) << "the SINGLE-sequence reference arm is not reproducible: " << d
                     << " of " << pool_elems << " floats differ between two identical runs";
    cudaFree(d_conv); cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_Alog);
    cudaFree(d_dtb); cudaFree(d_pool); cudaFree(d_y);
}

// Control 2: is the BATCHED arm reproducible against itself? Separates "race in
// the batched kernel" from "the comparison in run_and_compare is wrong".
TEST_F(GdnBatchedScanTest, BatchedArmIsReproducible) {
    const int n_seq = 32, n_tokens = 1, n_heads = 8, head_dim = 128, state_size = 128, n_groups = 8;
    const int conv_channels = 2 * n_groups * state_size + n_heads * head_dim;
    const int inner = n_heads * head_dim;
    const size_t rows = static_cast<size_t>(n_seq) * n_tokens;
    const size_t state_elems = static_cast<size_t>(n_heads) * state_size * head_dim;
    const size_t pool_elems = static_cast<size_t>(n_seq) * state_elems;
    std::vector<float> h_conv(rows*conv_channels), h_af(rows*n_heads), h_bf(rows*n_heads);
    std::vector<float> h_Alog(n_heads), h_dtb(n_heads), h_pool_init(pool_elems);
    fill(h_conv,1234); fill(h_af,5678,-2.f,2.f); fill(h_bf,9012,-2.f,2.f);
    fill(h_Alog,3456,-4.f,-0.5f); fill(h_dtb,7890,-1.f,1.f); fill(h_pool_init,2468,-0.5f,0.5f);
    std::vector<half> h_a(h_af.size()), h_b(h_bf.size());
    for (size_t i=0;i<h_af.size();i++) h_a[i]=__float2half(h_af[i]);
    for (size_t i=0;i<h_bf.size();i++) h_b[i]=__float2half(h_bf[i]);
    std::vector<int> slots(n_seq); for (int i=0;i<n_seq;i++) slots[i]=i;

    float *d_conv,*d_Alog,*d_dtb,*d_pool; half *d_alpha,*d_beta,*d_y; int* d_slots;
    ASSERT_EQ(cudaMalloc(&d_conv,h_conv.size()*sizeof(float)),cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_alpha,h_a.size()*sizeof(half)),cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_beta,h_b.size()*sizeof(half)),cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_Alog,n_heads*sizeof(float)),cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dtb,n_heads*sizeof(float)),cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pool,pool_elems*sizeof(float)),cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y,rows*inner*sizeof(half)),cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_slots,slots.size()*sizeof(int)),cudaSuccess);
    cudaMemcpy(d_conv,h_conv.data(),h_conv.size()*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha,h_a.data(),h_a.size()*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,h_b.data(),h_b.size()*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Alog,h_Alog.data(),n_heads*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_dtb,h_dtb.data(),n_heads*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_slots,slots.data(),slots.size()*sizeof(int),cudaMemcpyHostToDevice);

    std::vector<std::vector<float>> pools(2);
    for (int rep=0; rep<2; rep++) {
        cudaMemcpy(d_pool,h_pool_init.data(),pool_elems*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemset(d_y,0,rows*inner*sizeof(half));
        gdn_scan_fused_f32_batched(d_conv,conv_channels,d_alpha,d_beta,d_Alog,d_dtb,d_pool,d_slots,
                                   static_cast<int64_t>(state_elems),d_y,n_seq,n_tokens,n_heads,
                                   head_dim,state_size,n_groups,nullptr,1);
        ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
        pools[rep].resize(pool_elems);
        cudaMemcpy(pools[rep].data(),d_pool,pool_elems*sizeof(float),cudaMemcpyDeviceToHost);
    }
    size_t d=0; for (size_t i=0;i<pool_elems;i++) if (pools[0][i]!=pools[1][i]) d++;
    EXPECT_EQ(d,0u) << "BATCHED arm not reproducible: " << d << " of " << pool_elems << " differ";
    cudaFree(d_conv);cudaFree(d_alpha);cudaFree(d_beta);cudaFree(d_Alog);
    cudaFree(d_dtb);cudaFree(d_pool);cudaFree(d_y);cudaFree(d_slots);
}

TEST_F(GdnBatchedScanTest, SingleSequenceMatchesTheUnbatchedLauncher) {
    // n_seq=1 must be the shipped path exactly: gridDim.y == 1, every offset 0.
    run_and_compare(ScanShape{/*n_seq=*/1, /*n_tokens=*/1}, {0});
}

TEST_F(GdnBatchedScanTest, EightDecodeSequencesAreBitIdentical) {
    // The decode shape: one token per sequence, eight sequences at once.
    run_and_compare(ScanShape{8, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

TEST_F(GdnBatchedScanTest, ThirtyTwoDecodeSequencesAreBitIdentical) {
    ScanShape s{32, 1};
    std::vector<int> slots(32);
    for (int i = 0; i < 32; i++)
        slots[i] = i;
    run_and_compare(s, slots);
}

TEST_F(GdnBatchedScanTest, SlotsNeedNotBeContiguousOrOrdered) {
    // Real slot ids come from the scheduler's free list: sparse and unordered.
    // A kernel that assumed slot == blockIdx.y would pass every test above and
    // fail here, having written each sequence's state into another's slot.
    run_and_compare(ScanShape{4, 1}, {7, 2, 11, 0});
}

TEST_F(GdnBatchedScanTest, MultiTokenRowsStayCausalPerSequence) {
    // Not a decode shape, but it pins the invariant that matters: within a
    // sequence the scan is still sequential over tokens, and batching must not
    // leak row t of one sequence into row t of another.
    run_and_compare(ScanShape{4, 6}, {3, 1, 2, 0});
}

// The scan is useless batched if the depthwise causal conv in front of it still runs one
// sequence per launch - both are per-sequence state.

TEST_F(GdnBatchedScanTest, Conv1dDecodeBatchedMatchesPerSequence) {
    const int n_seq = 32, channels = 10240, ksize = 4;
    const size_t state_elems = static_cast<size_t>(channels) * ksize;
    const size_t pool_elems = static_cast<size_t>(n_seq) * state_elems;

    std::vector<float> h_state_init(pool_elems), h_w(state_elems), h_b(channels);
    std::vector<float> h_xf(static_cast<size_t>(n_seq) * channels);
    fill(h_state_init, 11, -1.0f, 1.0f);
    fill(h_w, 22, -0.5f, 0.5f);
    fill(h_b, 33, -0.2f, 0.2f);
    fill(h_xf, 44, -2.0f, 2.0f);
    std::vector<half> h_wh(state_elems), h_bh(channels), h_x(h_xf.size());
    for (size_t i = 0; i < state_elems; i++) h_wh[i] = __float2half(h_w[i]);
    for (int i = 0; i < channels; i++) h_bh[i] = __float2half(h_b[i]);
    for (size_t i = 0; i < h_xf.size(); i++) h_x[i] = __float2half(h_xf[i]);

    // Slots deliberately sparse and unordered, as the scheduler's free list is.
    std::vector<int> slots(n_seq);
    for (int i = 0; i < n_seq; i++) slots[i] = (n_seq - 1) - i;

    float *d_pool = nullptr, *d_out = nullptr;
    half *d_w = nullptr, *d_b = nullptr, *d_x = nullptr;
    int* d_slots = nullptr;
    ASSERT_EQ(cudaMalloc(&d_pool, pool_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, h_xf.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, state_elems * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, channels * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, h_x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_slots, n_seq * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_w, h_wh.data(), state_elems * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_bh.data(), channels * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slots, slots.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);

    Tensor w{}; w.data = d_w; w.ndim = 2; w.shape[0] = channels; w.shape[1] = ksize; w.qtype = QType::F16;
    Tensor b{}; b.data = d_b; b.ndim = 1; b.shape[0] = channels; b.qtype = QType::F16;
    Tensor x1{}; x1.ndim = 1; x1.shape[0] = channels; x1.qtype = QType::F16;

    // arm A: per sequence
    cudaMemcpy(d_pool, h_state_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, h_xf.size() * sizeof(float));
    for (int i = 0; i < n_seq; i++) {
        x1.data = d_x + static_cast<size_t>(i) * channels;
        ssm_conv1d_decode_f32_silu(d_pool + static_cast<size_t>(slots[i]) * state_elems, x1, w, b,
                                   d_out + static_cast<size_t>(i) * channels, ksize, nullptr);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> pool_a(pool_elems), out_a(h_xf.size());
    cudaMemcpy(pool_a.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_a.data(), d_out, out_a.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // arm B: one launch
    cudaMemcpy(d_pool, h_state_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, h_xf.size() * sizeof(float));
    ssm_conv1d_decode_f32_silu_batched(d_pool, d_slots, static_cast<int64_t>(state_elems), d_x, w, b,
                                       d_out, n_seq, channels, ksize, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> pool_b(pool_elems), out_b(h_xf.size());
    cudaMemcpy(pool_b.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b.data(), d_out, out_b.size() * sizeof(float), cudaMemcpyDeviceToHost);

    size_t sd = 0, od = 0;
    for (size_t i = 0; i < pool_elems; i++) if (pool_a[i] != pool_b[i]) sd++;
    for (size_t i = 0; i < out_a.size(); i++) if (out_a[i] != out_b[i]) od++;
    EXPECT_EQ(sd, 0u) << "conv state differs in " << sd << " of " << pool_elems;
    EXPECT_EQ(od, 0u) << "conv output differs in " << od << " of " << out_a.size();

    cudaFree(d_pool); cudaFree(d_out); cudaFree(d_w); cudaFree(d_b); cudaFree(d_x); cudaFree(d_slots);
}

// Grouped verify chunk (multi-candidate speculation on a hybrid): W candidates as W uniform
// sequences of T rows, each group committing state at d_real_n rows (pads past it only
// define discarded y), row-0 snapshot from group 0 only. Reference: the shipped
// single-sequence launcher run on the first real_n rows of each group.

TEST_F(GdnBatchedScanTest, GroupedChunkCommitsAtRealRowAndSnapshotsGroupZero) {
    const ScanShape s{/*n_seq=*/3, /*n_tokens=*/5};
    const std::vector<int> slots{4, 1, 2};
    const int real_n = 3, snap_rows = 1;
    const int conv_channels = 2 * s.n_groups * s.state_size + s.n_heads * s.head_dim;
    const int inner = s.n_heads * s.head_dim;
    const size_t rows = static_cast<size_t>(s.n_seq) * s.n_tokens;
    const size_t state_elems = static_cast<size_t>(s.n_heads) * s.state_size * s.head_dim;
    const int max_slot = *std::max_element(slots.begin(), slots.end());
    const size_t pool_elems = static_cast<size_t>(max_slot + 1) * state_elems;

    std::vector<float> h_conv(rows * conv_channels), h_alpha_f(rows * s.n_heads), h_beta_f(rows * s.n_heads);
    std::vector<float> h_Alog(s.n_heads), h_dtb(s.n_heads), h_pool_init(pool_elems);
    fill(h_conv, 4321);
    fill(h_alpha_f, 8765, -2.0f, 2.0f);
    fill(h_beta_f, 2109, -2.0f, 2.0f);
    fill(h_Alog, 6543, -4.0f, -0.5f);
    fill(h_dtb, 987, -1.0f, 1.0f);
    fill(h_pool_init, 1357, -0.5f, 0.5f);
    std::vector<half> h_alpha(h_alpha_f.size()), h_beta(h_beta_f.size());
    for (size_t i = 0; i < h_alpha_f.size(); i++) h_alpha[i] = __float2half(h_alpha_f[i]);
    for (size_t i = 0; i < h_beta_f.size(); i++) h_beta[i] = __float2half(h_beta_f[i]);

    float *d_conv = nullptr, *d_Alog = nullptr, *d_dtb = nullptr, *d_pool = nullptr, *d_snap = nullptr;
    half *d_alpha = nullptr, *d_beta = nullptr, *d_y = nullptr;
    int *d_slots = nullptr, *d_lens = nullptr;
    ASSERT_EQ(cudaMalloc(&d_conv, h_conv.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_alpha, h_alpha.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_beta, h_beta.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_Alog, h_Alog.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dtb, h_dtb.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pool, pool_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_snap, state_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, rows * inner * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_slots, slots.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_lens, 2 * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha.data(), h_alpha.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Alog, h_Alog.data(), h_Alog.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dtb, h_dtb.data(), h_dtb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slots, slots.data(), slots.size() * sizeof(int), cudaMemcpyHostToDevice);
    const int lens[2] = {real_n, snap_rows};
    cudaMemcpy(d_lens, lens, sizeof(lens), cudaMemcpyHostToDevice);

    auto seq_ptr = [&](int i, size_t per_row) { return static_cast<size_t>(i) * s.n_tokens * per_row; };

    // ---- reference: per group, the single launcher over its first real_n rows
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < s.n_seq; i++) {
        gdn_scan_fused_f32(d_conv + seq_ptr(i, conv_channels), conv_channels, d_alpha + seq_ptr(i, s.n_heads),
                           d_beta + seq_ptr(i, s.n_heads), d_Alog, d_dtb,
                           d_pool + static_cast<size_t>(slots[i]) * state_elems, d_y + seq_ptr(i, inner),
                           real_n, s.n_heads, s.head_dim, s.state_size, s.n_groups, nullptr,
                           /*grouped_layout=*/1);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> pool_ref(pool_elems);
    cudaMemcpy(pool_ref.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    // ... and group 0 after one row: the snapshot's reference.
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    gdn_scan_fused_f32(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb,
                       d_pool + static_cast<size_t>(slots[0]) * state_elems, d_y, snap_rows, s.n_heads,
                       s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> snap_ref(state_elems);
    cudaMemcpy(snap_ref.data(), d_pool + static_cast<size_t>(slots[0]) * state_elems,
               state_elems * sizeof(float), cudaMemcpyDeviceToHost);

    // ---- grouped: one launch over all n_tokens rows, committed at real_n
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_snap, 0, state_elems * sizeof(float));
    cudaMemset(d_y, 0, rows * inner * sizeof(half));
    gdn_scan_fused_f32_batched(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb, d_pool, d_slots,
                               static_cast<int64_t>(state_elems), d_y, s.n_seq, s.n_tokens, s.n_heads,
                               s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1,
                               /*d_real_n=*/d_lens, /*seq_row_offsets=*/nullptr, d_snap, d_lens + 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> pool_grp(pool_elems), snap_grp(state_elems);
    std::vector<half> y_grp(rows * inner);
    cudaMemcpy(pool_grp.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(snap_grp.data(), d_snap, state_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_grp.data(), d_y, y_grp.size() * sizeof(half), cudaMemcpyDeviceToHost);

    auto rms = [](const std::vector<float>& v) {
        double ss = 0.0;
        for (float x : v) ss += double(x) * double(x);
        return std::max(1e-9, std::sqrt(ss / static_cast<double>(v.size())));
    };
    auto worst = [](const std::vector<float>& a, const std::vector<float>& b, size_t off, size_t n) {
        double w = 0.0;
        for (size_t i = off; i < off + n; i++) w = std::max(w, std::abs(double(a[i]) - double(b[i])));
        return w;
    };
    const double scale = rms(pool_ref);
    // Every group's committed state is the state after real_n rows - the pads
    // advanced nothing (SPLIT=2 rounding tolerance, see the header comment).
    for (int i = 0; i < s.n_seq; i++) {
        const size_t off = static_cast<size_t>(slots[i]) * state_elems;
        EXPECT_LT(worst(pool_ref, pool_grp, off, state_elems) / scale, 1e-3)
            << "group " << i << " (slot " << slots[i] << ") did not commit at real_n=" << real_n;
    }
    // Slots no group owns are untouched.
    for (int sl = 0; sl <= max_slot; sl++) {
        if (std::find(slots.begin(), slots.end(), sl) != slots.end()) continue;
        EXPECT_EQ(worst(h_pool_init, pool_grp, static_cast<size_t>(sl) * state_elems, state_elems), 0.0)
            << "unowned slot " << sl << " was written";
    }
    // The snapshot is group 0 after snap_rows rows - and it moved.
    EXPECT_LT(worst(snap_ref, snap_grp, 0, state_elems) / scale, 1e-3) << "snapshot is not group 0's row-0 state";
    size_t snap_nonzero = 0;
    for (float x : snap_grp) if (x != 0.0f) snap_nonzero++;
    EXPECT_GT(snap_nonzero, state_elems / 4) << "snapshot slab was not written";
    // Pad rows still get a finite y (their output is discarded, not read as NaN).
    size_t y_nonfinite = 0;
    for (const half& v : y_grp) if (!std::isfinite(__half2float(v))) y_nonfinite++;
    EXPECT_EQ(y_nonfinite, 0u);

    cudaFree(d_conv); cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_Alog); cudaFree(d_dtb);
    cudaFree(d_pool); cudaFree(d_snap); cudaFree(d_y); cudaFree(d_slots); cudaFree(d_lens);
}

TEST_F(GdnBatchedScanTest, GroupedConvCommitsPerSlotAndSnapshotsGroupZero) {
    // Conv half of the grouped verify chunk: n_seq groups of n_tokens rows on the window of
    // slot seq_slots[z], committed at real_n rows; snapshot (window after snap_rows rows,
    // leading values from the PRE-chunk copy) from group 0 only. Reference: single-sequence
    // prefill conv per group over its first real_n rows.
    const int n_seq = 3, n_tokens = 5, real_n = 3, snap_rows = 1, channels = 2048, ksize = 4;
    const std::vector<int> slots{4, 1, 2};
    const int max_slot = 4;
    const size_t win = static_cast<size_t>(channels) * ksize;
    const size_t pool_elems = static_cast<size_t>(max_slot + 1) * win;
    const size_t rows = static_cast<size_t>(n_seq) * n_tokens;

    std::vector<float> h_pool_init(pool_elems), h_w(win), h_b(channels), h_xf(rows * channels);
    fill(h_pool_init, 101, -1.0f, 1.0f);
    fill(h_w, 202, -0.5f, 0.5f);
    fill(h_b, 303, -0.2f, 0.2f);
    fill(h_xf, 404, -2.0f, 2.0f);
    std::vector<half> h_wh(win), h_bh(channels), h_x(h_xf.size());
    for (size_t i = 0; i < win; i++) h_wh[i] = __float2half(h_w[i]);
    for (int i = 0; i < channels; i++) h_bh[i] = __float2half(h_b[i]);
    for (size_t i = 0; i < h_xf.size(); i++) h_x[i] = __float2half(h_xf[i]);

    float *d_pool = nullptr, *d_out = nullptr, *d_snap = nullptr, *d_prev = nullptr;
    half *d_w = nullptr, *d_b = nullptr, *d_x = nullptr;
    int *d_slots = nullptr, *d_lens = nullptr;
    ASSERT_EQ(cudaMalloc(&d_pool, pool_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, rows * channels * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_snap, win * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_prev, win * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, win * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, channels * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, h_x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_slots, n_seq * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_lens, 2 * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_w, h_wh.data(), win * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_bh.data(), channels * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slots, slots.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    const int lens[2] = {real_n, snap_rows};
    cudaMemcpy(d_lens, lens, sizeof(lens), cudaMemcpyHostToDevice);
    // The pre-chunk copy of group 0's window (the engine's spec_state_scratch_).
    cudaMemcpy(d_prev, h_pool_init.data() + static_cast<size_t>(slots[0]) * win, win * sizeof(float),
               cudaMemcpyHostToDevice);

    Tensor w{}; w.data = d_w; w.ndim = 2; w.shape[0] = channels; w.shape[1] = ksize; w.qtype = QType::F16;
    Tensor b{}; b.data = d_b; b.ndim = 1; b.shape[0] = channels; b.qtype = QType::F16;

    // reference: per group, first real_n rows, snapshot from group 0
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, rows * channels * sizeof(float));
    cudaMemset(d_snap, 0, win * sizeof(float));
    for (int i = 0; i < n_seq; i++) {
        Tensor xs{}; xs.data = d_x + static_cast<size_t>(i) * n_tokens * channels; xs.ndim = 2;
        xs.shape[0] = real_n; xs.shape[1] = channels; xs.qtype = QType::F16;
        ssm_conv1d_prefill_f32_silu(d_pool + static_cast<size_t>(slots[i]) * win, xs, w, b,
                                    d_out + static_cast<size_t>(i) * n_tokens * channels, ksize, nullptr,
                                    /*d_real_n=*/nullptr, i == 0 ? d_snap : nullptr,
                                    i == 0 ? d_lens + 1 : nullptr, i == 0 ? d_prev : nullptr);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> pool_a(pool_elems), out_a(rows * channels), snap_a(win);
    cudaMemcpy(pool_a.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_a.data(), d_out, out_a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(snap_a.data(), d_snap, win * sizeof(float), cudaMemcpyDeviceToHost);

    // grouped: one launch over all n_tokens rows per group
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, rows * channels * sizeof(float));
    cudaMemset(d_snap, 0, win * sizeof(float));
    Tensor xg{}; xg.data = d_x; xg.ndim = 2; xg.shape[0] = static_cast<int64_t>(rows); xg.shape[1] = channels;
    xg.qtype = QType::F16;
    ssm_conv1d_prefill_f32_silu_grouped(d_pool, d_slots, static_cast<int64_t>(win), n_seq, xg, w, b, d_out,
                                        ksize, nullptr, d_lens, d_snap, d_lens + 1, d_prev);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> pool_b(pool_elems), out_b(rows * channels), snap_b(win);
    cudaMemcpy(pool_b.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b.data(), d_out, out_b.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(snap_b.data(), d_snap, win * sizeof(float), cudaMemcpyDeviceToHost);

    // Same arithmetic per element - bit-compare. Windows: every slot (owned
    // ones committed at real_n, unowned ones untouched); outputs: the first
    // real_n rows of every group; snapshot: group 0's.
    size_t sd = 0, od = 0, snd = 0, pad_nonfinite = 0;
    for (size_t i = 0; i < pool_elems; i++) if (pool_a[i] != pool_b[i]) sd++;
    for (int g = 0; g < n_seq; g++)
        for (size_t i = 0; i < static_cast<size_t>(real_n) * channels; i++) {
            const size_t k = static_cast<size_t>(g) * n_tokens * channels + i;
            if (out_a[k] != out_b[k]) od++;
        }
    for (size_t i = 0; i < win; i++) if (snap_a[i] != snap_b[i]) snd++;
    for (size_t i = 0; i < out_b.size(); i++) if (!std::isfinite(out_b[i])) pad_nonfinite++;
    EXPECT_EQ(sd, 0u) << "conv windows differ in " << sd << " of " << pool_elems;
    EXPECT_EQ(od, 0u) << "conv outputs differ in " << od << " real-row elements";
    EXPECT_EQ(snd, 0u) << "group-0 snapshot differs in " << snd << " of " << win;
    EXPECT_EQ(pad_nonfinite, 0u);
    size_t snap_moved = 0;
    for (size_t i = 0; i < win; i++) if (snap_b[i] != h_pool_init[static_cast<size_t>(slots[0]) * win + i]) snap_moved++;
    EXPECT_GT(snap_moved, 0u) << "snapshot window did not move";

    cudaFree(d_pool); cudaFree(d_out); cudaFree(d_snap); cudaFree(d_prev); cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_x); cudaFree(d_slots); cudaFree(d_lens);
}

// Batched speculative verify geometry: N groups of 2 rows (last token + draft), group z
// reads slot L_z, commits the 2-row state into spare slot P_z and the 1-row state in place
// into L_z. Reference: the single launcher per group over 2 rows (->P_z) and 1 row (->L_z).
TEST_F(GdnBatchedScanTest, VerifyGroupsCommitToSpareAndSnapshotInPlace) {
    const ScanShape s{/*n_seq=*/4, /*n_tokens=*/2};
    const std::vector<int> live{5, 1, 3, 0};
    const std::vector<int> spare{2, 7, 4, 6};
    const int conv_channels = 2 * s.n_groups * s.state_size + s.n_heads * s.head_dim;
    const int inner = s.n_heads * s.head_dim;
    const size_t rows = static_cast<size_t>(s.n_seq) * s.n_tokens;
    const size_t state_elems = static_cast<size_t>(s.n_heads) * s.state_size * s.head_dim;
    const int n_slots = 8;
    const size_t pool_elems = static_cast<size_t>(n_slots) * state_elems;

    std::vector<float> h_conv(rows * conv_channels), h_alpha_f(rows * s.n_heads), h_beta_f(rows * s.n_heads);
    std::vector<float> h_Alog(s.n_heads), h_dtb(s.n_heads), h_pool_init(pool_elems);
    fill(h_conv, 7431);
    fill(h_alpha_f, 1865, -2.0f, 2.0f);
    fill(h_beta_f, 9210, -2.0f, 2.0f);
    fill(h_Alog, 3654, -4.0f, -0.5f);
    fill(h_dtb, 789, -1.0f, 1.0f);
    fill(h_pool_init, 5317, -0.5f, 0.5f);
    std::vector<half> h_alpha(h_alpha_f.size()), h_beta(h_beta_f.size());
    for (size_t i = 0; i < h_alpha_f.size(); i++) h_alpha[i] = __float2half(h_alpha_f[i]);
    for (size_t i = 0; i < h_beta_f.size(); i++) h_beta[i] = __float2half(h_beta_f[i]);

    float *d_conv = nullptr, *d_Alog = nullptr, *d_dtb = nullptr, *d_pool = nullptr;
    half *d_alpha = nullptr, *d_beta = nullptr, *d_y = nullptr;
    int *d_live = nullptr, *d_spare = nullptr, *d_lens = nullptr;
    ASSERT_EQ(cudaMalloc(&d_conv, h_conv.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_alpha, h_alpha.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_beta, h_beta.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_Alog, h_Alog.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dtb, h_dtb.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pool, pool_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, rows * inner * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_live, live.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_spare, spare.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_lens, 2 * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha.data(), h_alpha.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Alog, h_Alog.data(), h_Alog.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dtb, h_dtb.data(), h_dtb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_live, live.data(), live.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spare, spare.data(), spare.size() * sizeof(int), cudaMemcpyHostToDevice);
    const int lens[2] = {2, 1};
    cudaMemcpy(d_lens, lens, sizeof(lens), cudaMemcpyHostToDevice);

    auto seq_ptr = [&](int i, size_t per_row) { return static_cast<size_t>(i) * s.n_tokens * per_row; };
    auto slot = [&](std::vector<float>& p, int sl) { return p.data() + static_cast<size_t>(sl) * state_elems; };

    // ---- reference: per group, 2 rows from L_z -> expected P_z; 1 row from L_z -> expected L_z
    std::vector<float> want(pool_elems);
    std::copy(h_pool_init.begin(), h_pool_init.end(), want.begin());
    for (int i = 0; i < s.n_seq; i++) {
        for (int n = 1; n <= 2; n++) {
            cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
            gdn_scan_fused_f32(d_conv + seq_ptr(i, conv_channels), conv_channels, d_alpha + seq_ptr(i, s.n_heads),
                               d_beta + seq_ptr(i, s.n_heads), d_Alog, d_dtb,
                               d_pool + static_cast<size_t>(live[i]) * state_elems, d_y + seq_ptr(i, inner), n,
                               s.n_heads, s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            cudaMemcpy(slot(want, n == 2 ? spare[i] : live[i]), d_pool + static_cast<size_t>(live[i]) * state_elems,
                       state_elems * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    // ---- batched: one launch, out_slots = spare, snap_slots = live (in place), snap at 1 row
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, rows * inner * sizeof(half));
    gdn_scan_fused_f32_batched(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb, d_pool, d_live,
                               static_cast<int64_t>(state_elems), d_y, s.n_seq, s.n_tokens, s.n_heads,
                               s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1,
                               /*d_real_n=*/d_lens, /*seq_row_offsets=*/nullptr, /*h_snap=*/nullptr,
                               /*d_snap_n=*/d_lens + 1, /*out_slots=*/d_spare, /*snap_slots=*/d_live);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> got(pool_elems);
    cudaMemcpy(got.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);

    auto rms = [](const std::vector<float>& v) {
        double ss = 0.0;
        for (float x : v) ss += double(x) * double(x);
        return std::max(1e-9, std::sqrt(ss / static_cast<double>(v.size())));
    };
    auto worst = [](const std::vector<float>& a, const std::vector<float>& b, size_t off, size_t n) {
        double w = 0.0;
        for (size_t i = off; i < off + n; i++) w = std::max(w, std::abs(double(a[i]) - double(b[i])));
        return w;
    };
    const double scale = rms(want);
    for (int i = 0; i < s.n_seq; i++) {
        const size_t off_p = static_cast<size_t>(spare[i]) * state_elems;
        const size_t off_l = static_cast<size_t>(live[i]) * state_elems;
        EXPECT_LT(worst(want, got, off_p, state_elems) / scale, 1e-3)
            << "group " << i << ": spare slot " << spare[i] << " is not the 2-row state";
        EXPECT_LT(worst(want, got, off_l, state_elems) / scale, 1e-3)
            << "group " << i << ": live slot " << live[i] << " is not the 1-row state";
        // The two committed states differ (row 1 moved the state).
        double moved = 0.0;
        for (size_t k = 0; k < state_elems; k++)
            moved = std::max(moved, std::abs(double(got[off_p + k]) - double(got[off_l + k])));
        EXPECT_GT(moved / scale, 1e-3) << "group " << i << ": spare and live slots hold the same state";
    }
    // No slot outside live+spare was written.
    for (int sl = 0; sl < n_slots; sl++) {
        if (std::find(live.begin(), live.end(), sl) != live.end() ||
            std::find(spare.begin(), spare.end(), sl) != spare.end())
            continue;
        EXPECT_EQ(worst(h_pool_init, got, static_cast<size_t>(sl) * state_elems, state_elems), 0.0)
            << "unowned slot " << sl << " was written";
    }

    cudaFree(d_conv); cudaFree(d_alpha); cudaFree(d_beta); cudaFree(d_Alog); cudaFree(d_dtb);
    cudaFree(d_pool); cudaFree(d_y); cudaFree(d_live); cudaFree(d_spare); cudaFree(d_lens);
}

// The conv half of the same geometry: window at 2 rows -> spare slot, window
// at 1 row in place. Reference: the single launcher per group and row count.
TEST_F(GdnBatchedScanTest, VerifyGroupsConvCommitsToSpareAndSnapshotsInPlace) {
    const int n_seq = 4, n_tokens = 2, channels = 2048, ksize = 4, n_slots = 8;
    const std::vector<int> live{5, 1, 3, 0};
    const std::vector<int> spare{2, 7, 4, 6};
    const size_t win = static_cast<size_t>(channels) * ksize;
    const size_t pool_elems = static_cast<size_t>(n_slots) * win;
    const size_t rows = static_cast<size_t>(n_seq) * n_tokens;

    std::vector<float> h_pool_init(pool_elems), h_w(win), h_b(channels), h_xf(rows * channels);
    fill(h_pool_init, 111, -1.0f, 1.0f);
    fill(h_w, 222, -0.5f, 0.5f);
    fill(h_b, 333, -0.2f, 0.2f);
    fill(h_xf, 444, -2.0f, 2.0f);
    std::vector<half> h_wh(win), h_bh(channels), h_x(h_xf.size());
    for (size_t i = 0; i < win; i++) h_wh[i] = __float2half(h_w[i]);
    for (int i = 0; i < channels; i++) h_bh[i] = __float2half(h_b[i]);
    for (size_t i = 0; i < h_xf.size(); i++) h_x[i] = __float2half(h_xf[i]);

    float *d_pool = nullptr, *d_out = nullptr;
    half *d_w = nullptr, *d_b = nullptr, *d_x = nullptr;
    int *d_live = nullptr, *d_spare = nullptr, *d_lens = nullptr;
    ASSERT_EQ(cudaMalloc(&d_pool, pool_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, rows * channels * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, win * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, channels * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, h_x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_live, n_seq * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_spare, n_seq * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_lens, 2 * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_w, h_wh.data(), win * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_bh.data(), channels * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_live, live.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spare, spare.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    const int lens[2] = {2, 1};
    cudaMemcpy(d_lens, lens, sizeof(lens), cudaMemcpyHostToDevice);

    Tensor w{}; w.data = d_w; w.ndim = 2; w.shape[0] = channels; w.shape[1] = ksize; w.qtype = QType::F16;
    Tensor b{}; b.data = d_b; b.ndim = 1; b.shape[0] = channels; b.qtype = QType::F16;

    // reference windows: per group, 2 rows -> spare, 1 row -> live; outputs from the 2-row run
    std::vector<float> want(pool_elems), out_a(rows * channels);
    std::copy(h_pool_init.begin(), h_pool_init.end(), want.begin());
    for (int i = 0; i < n_seq; i++) {
        for (int n = 1; n <= 2; n++) {
            cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
            Tensor xs{}; xs.data = d_x + static_cast<size_t>(i) * n_tokens * channels; xs.ndim = 2;
            xs.shape[0] = n; xs.shape[1] = channels; xs.qtype = QType::F16;
            ssm_conv1d_prefill_f32_silu(d_pool + static_cast<size_t>(live[i]) * win, xs, w, b,
                                        d_out + static_cast<size_t>(i) * n_tokens * channels, ksize, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            cudaMemcpy(want.data() + static_cast<size_t>(n == 2 ? spare[i] : live[i]) * win,
                       d_pool + static_cast<size_t>(live[i]) * win, win * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    cudaMemcpy(out_a.data(), d_out, out_a.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // grouped: one launch, commit to spare, snapshot (1 row) in place
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, rows * channels * sizeof(float));
    Tensor xg{}; xg.data = d_x; xg.ndim = 2; xg.shape[0] = static_cast<int64_t>(rows); xg.shape[1] = channels;
    xg.qtype = QType::F16;
    ssm_conv1d_prefill_f32_silu_grouped(d_pool, d_live, static_cast<int64_t>(win), n_seq, xg, w, b, d_out, ksize,
                                        nullptr, d_lens, /*conv_snap=*/nullptr, d_lens + 1, /*conv_prev=*/nullptr,
                                        /*out_slots=*/d_spare, /*snap_slots=*/d_live);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> got(pool_elems), out_b(rows * channels);
    cudaMemcpy(got.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b.data(), d_out, out_b.size() * sizeof(float), cudaMemcpyDeviceToHost);

    size_t wd = 0, od = 0;
    for (size_t i = 0; i < pool_elems; i++) if (want[i] != got[i]) wd++;
    for (size_t i = 0; i < out_a.size(); i++) if (out_a[i] != out_b[i]) od++;
    EXPECT_EQ(wd, 0u) << "conv windows differ in " << wd << " of " << pool_elems;
    EXPECT_EQ(od, 0u) << "conv outputs differ in " << od << " elements";
    size_t moved = 0;
    for (int i = 0; i < n_seq; i++)
        for (size_t k = 0; k < win; k++)
            if (got[static_cast<size_t>(spare[i]) * win + k] != got[static_cast<size_t>(live[i]) * win + k]) moved++;
    EXPECT_GT(moved, 0u) << "spare and live windows are identical";

    cudaFree(d_pool); cudaFree(d_out); cudaFree(d_w); cudaFree(d_b); cudaFree(d_x);
    cudaFree(d_live); cudaFree(d_spare); cudaFree(d_lens);
}


// Factored spare (compute/gdn_factor.cuh): the verify used to carry the drafted row as a
// second FULL state slot (an exact duplicate of the pool); the delta rule's per-token update
// is g*H + k(x)delta, so the row is a (g,k,delta) triple instead (48KiB vs 1.5MiB/layer on
// Qwen3.8-27B). Runs the real accept sequence both ways and compares state AFTER the
// following token, not the factors themselves - what must hold is indistinguishability.
TEST_F(GdnBatchedScanTest, FactoredSpareReproducesTheFullSpareAfterTheNextToken) {
    const ScanShape s{/*n_seq=*/3, /*n_tokens=*/2};
    const std::vector<int> live{4, 0, 2};
    const std::vector<int> spare{1, 5, 3};
    const int conv_channels = 2 * s.n_groups * s.state_size + s.n_heads * s.head_dim;
    const int inner = s.n_heads * s.head_dim;
    const size_t rows = static_cast<size_t>(s.n_seq) * s.n_tokens;
    const size_t state_elems = static_cast<size_t>(s.n_heads) * s.state_size * s.head_dim;
    const int n_slots = 6;
    const size_t pool_elems = static_cast<size_t>(n_slots) * state_elems;
    const int fac_stride = imp::gdn_factor_floats_per_head(s.state_size, s.head_dim);
    // Rows are indexed by SLOT, like the state pool, so the buffer spans slots.
    const size_t fac_elems = static_cast<size_t>(n_slots) * s.n_heads * fac_stride;

    std::vector<float> h_conv(rows * conv_channels), h_alpha_f(rows * s.n_heads), h_beta_f(rows * s.n_heads);
    std::vector<float> h_Alog(s.n_heads), h_dtb(s.n_heads), h_pool_init(pool_elems);
    // A third token per sequence: the step that follows the accept.
    std::vector<float> h_conv3(static_cast<size_t>(s.n_seq) * conv_channels);
    std::vector<float> h_alpha3_f(static_cast<size_t>(s.n_seq) * s.n_heads);
    std::vector<float> h_beta3_f(static_cast<size_t>(s.n_seq) * s.n_heads);
    fill(h_conv, 2281);
    fill(h_alpha_f, 4409, -2.0f, 2.0f);
    fill(h_beta_f, 1123, -2.0f, 2.0f);
    fill(h_Alog, 6653, -4.0f, -0.5f);
    fill(h_dtb, 331, -1.0f, 1.0f);
    fill(h_pool_init, 8819, -0.5f, 0.5f);
    fill(h_conv3, 5501);
    fill(h_alpha3_f, 7717, -2.0f, 2.0f);
    fill(h_beta3_f, 9931, -2.0f, 2.0f);
    auto to_half = [](const std::vector<float>& f) {
        std::vector<half> h(f.size());
        for (size_t i = 0; i < f.size(); i++) h[i] = __float2half(f[i]);
        return h;
    };
    const std::vector<half> h_alpha = to_half(h_alpha_f), h_beta = to_half(h_beta_f);
    const std::vector<half> h_alpha3 = to_half(h_alpha3_f), h_beta3 = to_half(h_beta3_f);

    float *d_conv = nullptr, *d_conv3 = nullptr, *d_Alog = nullptr, *d_dtb = nullptr, *d_pool = nullptr;
    float* d_fac = nullptr;
    half *d_alpha = nullptr, *d_beta = nullptr, *d_alpha3 = nullptr, *d_beta3 = nullptr, *d_y = nullptr;
    int *d_live = nullptr, *d_spare = nullptr, *d_lens = nullptr;
    auto dev = [](auto** p, size_t bytes) { ASSERT_EQ(cudaMalloc(p, bytes), cudaSuccess); };
    dev(&d_conv, h_conv.size() * sizeof(float));
    dev(&d_conv3, h_conv3.size() * sizeof(float));
    dev(&d_alpha, h_alpha.size() * sizeof(half));
    dev(&d_beta, h_beta.size() * sizeof(half));
    dev(&d_alpha3, h_alpha3.size() * sizeof(half));
    dev(&d_beta3, h_beta3.size() * sizeof(half));
    dev(&d_Alog, h_Alog.size() * sizeof(float));
    dev(&d_dtb, h_dtb.size() * sizeof(float));
    dev(&d_pool, pool_elems * sizeof(float));
    dev(&d_fac, fac_elems * sizeof(float));
    dev(&d_y, rows * inner * sizeof(half));
    dev(&d_live, live.size() * sizeof(int));
    dev(&d_spare, spare.size() * sizeof(int));
    dev(&d_lens, 2 * sizeof(int));
    auto up = [](void* d, const auto& v) {
        cudaMemcpy(d, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice);
    };
    up(d_conv, h_conv); up(d_conv3, h_conv3); up(d_alpha, h_alpha); up(d_beta, h_beta);
    up(d_alpha3, h_alpha3); up(d_beta3, h_beta3); up(d_Alog, h_Alog); up(d_dtb, h_dtb);
    up(d_live, live); up(d_spare, spare);
    const int lens[2] = {2, 1};
    cudaMemcpy(d_lens, lens, sizeof(lens), cudaMemcpyHostToDevice);

    // Arm A, today's contract: the verify commits the drafted row into the
    // spare slot, accept makes the spare live, the next token runs from it.
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    gdn_scan_fused_f32_batched(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb, d_pool, d_live,
                               static_cast<int64_t>(state_elems), d_y, s.n_seq, s.n_tokens, s.n_heads,
                               s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1,
                               d_lens, nullptr, nullptr, d_lens + 1, /*out_slots=*/d_spare,
                               /*snap_slots=*/d_live);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    gdn_scan_fused_f32_batched(d_conv3, conv_channels, d_alpha3, d_beta3, d_Alog, d_dtb, d_pool, d_spare,
                               static_cast<int64_t>(state_elems), d_y, s.n_seq, /*n_tokens=*/1, s.n_heads,
                               s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> want(pool_elems);
    cudaMemcpy(want.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);

    // Arm B, factored: no spare slot. The verify emits (g, k, delta) for the
    // drafted row and snapshots the accepted row in place; the next token runs
    // from the live slot with the row applied at the state load.
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_fac, 0, fac_elems * sizeof(float));
    gdn_scan_fused_f32_batched(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb, d_pool, d_live,
                               static_cast<int64_t>(state_elems), d_y, s.n_seq, s.n_tokens, s.n_heads,
                               s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1,
                               d_lens, nullptr, nullptr, d_lens + 1, /*out_slots=*/nullptr,
                               /*snap_slots=*/d_live, /*fac_out=*/d_fac, /*fac_in=*/nullptr, fac_stride);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // The sentinel has to be live: every head must carry a non-zero decay.
    std::vector<float> fac(fac_elems);
    cudaMemcpy(fac.data(), d_fac, fac_elems * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < s.n_seq; i++)
        for (int h = 0; h < s.n_heads; h++)
            ASSERT_GT(fac[(static_cast<size_t>(live[i]) * s.n_heads + h) * fac_stride], 0.0f)
                << "seq " << i << " head " << h << ": g is the 'no pending row' sentinel";
    gdn_scan_fused_f32_batched(d_conv3, conv_channels, d_alpha3, d_beta3, d_Alog, d_dtb, d_pool, d_live,
                               static_cast<int64_t>(state_elems), d_y, s.n_seq, /*n_tokens=*/1, s.n_heads,
                               s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1,
                               nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                               /*fac_out=*/nullptr, /*fac_in=*/d_fac, fac_stride);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> got(pool_elems);
    cudaMemcpy(got.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);

    // Same arithmetic in the same order on both arms, so F32 state makes this
    // exact. A tolerance here would hide a wrong k or delta index.
    for (int i = 0; i < s.n_seq; i++) {
        const size_t a = static_cast<size_t>(spare[i]) * state_elems;
        const size_t b = static_cast<size_t>(live[i]) * state_elems;
        for (size_t e = 0; e < state_elems; e++)
            ASSERT_EQ(want[a + e], got[b + e]) << "seq " << i << " elem " << e;
    }

    for (void* p : {(void*)d_conv, (void*)d_conv3, (void*)d_alpha, (void*)d_beta, (void*)d_alpha3,
                    (void*)d_beta3, (void*)d_Alog, (void*)d_dtb, (void*)d_pool, (void*)d_fac,
                    (void*)d_y, (void*)d_live, (void*)d_spare, (void*)d_lens})
        cudaFree(p);
}

}  // namespace
}  // namespace imp
