// Batched GDN scan: N independent sequences in one launch.
//
// Concurrent GDN decode used to run one sequence per step
// (`engine_scheduler.cpp`: "the recurrent scan kernels are single-sequence"),
// which forced the whole decode step — including the FFN and attention
// projections that are ordinary GEMMs — onto the M=1 path. Profiled 2026-08-24
// under 32-way load: CUTLASS GEMM was 1.0 % of GPU time on a GDN model against
// 71.8 % on a dense one, while M=1 GEMV kernels were ~82 % against ~7 %. The
// scan that genuinely cannot batch was 3.8 % of that profile.
//
// Tokens within a sequence really are sequential. Separate sequences are not:
// each owns its own recurrent-state slot and they share nothing but weights, so
// they parallelise across blockIdx.y exactly like heads do across blockIdx.x.
//
// The contract this asserts is the one that makes the change safe to wire in:
// running N sequences batched must produce BIT-IDENTICAL states and outputs to
// running them one at a time. Not "close" — identical, because it is the same
// arithmetic in the same order, only in different blocks.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/gdn.h"

#include <cstdint>
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

    // ---- compare, bit-exact
    size_t state_diffs = 0, y_diffs = 0;
    for (size_t i = 0; i < pool_elems; i++)
        if (pool_single[i] != pool_batched[i])
            state_diffs++;
    for (size_t i = 0; i < y_single.size(); i++)
        if (__half2float(y_single[i]) != __half2float(y_batched[i]))
            y_diffs++;

    EXPECT_EQ(state_diffs, 0u) << "recurrent state differs in " << state_diffs << " of " << pool_elems
                               << " floats (n_seq=" << s.n_seq << ", n_tokens=" << s.n_tokens << ")";
    EXPECT_EQ(y_diffs, 0u) << "scan output differs in " << y_diffs << " of " << y_single.size()
                           << " halfs (n_seq=" << s.n_seq << ", n_tokens=" << s.n_tokens << ")";

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

}  // namespace
}  // namespace imp
