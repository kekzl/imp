// Bound verify at 2+ drafts (speculative.mtp_batch_verify_k, docs/plans/2026-09-11-batched-mtp-verify.md):
// groups of D + 1 rows (last token + D drafts). Second spare (D = 2): 1 row in place into the live slot L_z,
// 2 rows into spare2 Q_z, 3 rows into spare P_z; the scan writes the 2-row state in a commit-only pass
// before the main pass (run_gdn, ssm_snap_rows 2), the conv takes a third commit launch (snap_rows 2).
// Reference: the single-sequence launchers per group and row count. Factor rows (mtp_batch_verify_factor_row,
// D = 2 .. 4): no spare2, drafts 1 .. D - 1 as factor rows / conv taps by group, folded into the live slot
// on a partial accept.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "compute/gdn.h"
#include "compute/gdn_factor.cuh"
#include "compute/ssm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <type_traits>
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

void fill(std::vector<float>& v, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& x : v)
        x = d(rng);
}

class GdnVerifyTwoDraftsTest : public ::testing::Test {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

TEST_F(GdnVerifyTwoDraftsTest, ScanCommitsOneTwoAndThreeRowStates) {
    const ScanShape s{/*n_seq=*/4, /*n_tokens=*/3};
    const std::vector<int> live{5, 1, 3, 0};
    const std::vector<int> spare{2, 7, 4, 6};
    const std::vector<int> spare2{9, 11, 8, 10};
    std::vector<int> snaps(live);
    snaps.insert(snaps.end(), spare2.begin(), spare2.end());
    const int conv_channels = 2 * s.n_groups * s.state_size + s.n_heads * s.head_dim;
    const int inner = s.n_heads * s.head_dim;
    const size_t rows = static_cast<size_t>(s.n_seq) * s.n_tokens;
    const size_t state_elems = static_cast<size_t>(s.n_heads) * s.state_size * s.head_dim;
    const int n_slots = 12;
    const size_t pool_elems = static_cast<size_t>(n_slots) * state_elems;

    std::vector<float> h_conv(rows * conv_channels), h_alpha_f(rows * s.n_heads), h_beta_f(rows * s.n_heads);
    std::vector<float> h_Alog(s.n_heads), h_dtb(s.n_heads), h_pool_init(pool_elems);
    fill(h_conv, 7433);
    fill(h_alpha_f, 1867, -2.0f, 2.0f);
    fill(h_beta_f, 9212, -2.0f, 2.0f);
    fill(h_Alog, 3656, -4.0f, -0.5f);
    fill(h_dtb, 791, -1.0f, 1.0f);
    fill(h_pool_init, 5319, -0.5f, 0.5f);
    std::vector<half> h_alpha(h_alpha_f.size()), h_beta(h_beta_f.size());
    for (size_t i = 0; i < h_alpha_f.size(); i++)
        h_alpha[i] = __float2half(h_alpha_f[i]);
    for (size_t i = 0; i < h_beta_f.size(); i++)
        h_beta[i] = __float2half(h_beta_f[i]);

    float *d_conv = nullptr, *d_Alog = nullptr, *d_dtb = nullptr, *d_pool = nullptr;
    half *d_alpha = nullptr, *d_beta = nullptr, *d_y = nullptr;
    int *d_live = nullptr, *d_spare = nullptr, *d_snaps = nullptr, *d_lens = nullptr;
    ASSERT_EQ(cudaMalloc(&d_conv, h_conv.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_alpha, h_alpha.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_beta, h_beta.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_Alog, h_Alog.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dtb, h_dtb.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pool, pool_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, rows * inner * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_live, live.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_spare, spare.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_snaps, snaps.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_lens, 3 * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha.data(), h_alpha.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Alog, h_Alog.data(), h_Alog.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dtb, h_dtb.data(), h_dtb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_live, live.data(), live.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spare, spare.data(), spare.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snaps, snaps.data(), snaps.size() * sizeof(int), cudaMemcpyHostToDevice);
    const int lens[3] = {3, 1, 2};
    cudaMemcpy(d_lens, lens, sizeof(lens), cudaMemcpyHostToDevice);

    auto seq_ptr = [&](int i, size_t per_row) { return static_cast<size_t>(i) * s.n_tokens * per_row; };
    auto slot = [&](std::vector<float>& p, int sl) {
        return p.data() + static_cast<size_t>(sl) * state_elems;
    };

    // ---- reference: per group, n rows from L_z -> L_z (1), Q_z (2), P_z (3)
    std::vector<float> want(pool_elems);
    std::copy(h_pool_init.begin(), h_pool_init.end(), want.begin());
    for (int i = 0; i < s.n_seq; i++) {
        for (int n = 1; n <= 3; n++) {
            cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
            gdn_scan_fused_f32(d_conv + seq_ptr(i, conv_channels), conv_channels,
                               d_alpha + seq_ptr(i, s.n_heads), d_beta + seq_ptr(i, s.n_heads), d_Alog, d_dtb,
                               d_pool + static_cast<size_t>(live[i]) * state_elems, d_y + seq_ptr(i, inner),
                               n, s.n_heads, s.head_dim, s.state_size, s.n_groups, nullptr,
                               /*grouped_layout=*/1);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const int dst = n == 1 ? live[i] : (n == 2 ? spare2[i] : spare[i]);
            cudaMemcpy(slot(want, dst), d_pool + static_cast<size_t>(live[i]) * state_elems,
                       state_elems * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    // ---- batched as run_gdn stages it: commit-only pass (2 rows -> Q_z), then the main pass
    // (3 rows -> P_z, 1 row in place). The main pass alone gives the reference y rows.
    auto main_pass = [&] {
        gdn_scan_fused_f32_batched(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb, d_pool, d_live,
                                   static_cast<int64_t>(state_elems), d_y, s.n_seq, s.n_tokens, s.n_heads,
                                   s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1,
                                   /*d_real_n=*/d_lens, /*seq_row_offsets=*/nullptr, /*h_snap=*/nullptr,
                                   /*d_snap_n=*/d_lens + 1, /*out_slots=*/d_spare, /*snap_slots=*/d_snaps);
    };
    std::vector<half> y_one(rows * inner), y_two(rows * inner);
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    main_pass();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(y_one.data(), d_y, y_one.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, rows * inner * sizeof(half));
    gdn_scan_fused_f32_batched(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb, d_pool, d_live,
                               static_cast<int64_t>(state_elems), d_y, s.n_seq, s.n_tokens, s.n_heads,
                               s.head_dim, s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1,
                               /*d_real_n=*/d_lens + 2, /*seq_row_offsets=*/nullptr, /*h_snap=*/nullptr,
                               /*d_snap_n=*/nullptr, /*out_slots=*/d_snaps + s.n_seq, /*snap_slots=*/nullptr);
    main_pass();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> got(pool_elems);
    cudaMemcpy(got.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_two.data(), d_y, y_two.size() * sizeof(half), cudaMemcpyDeviceToHost);
    size_t yd = 0;
    for (size_t k = 0; k < y_one.size(); k++)
        yd += __half_as_ushort(y_one[k]) != __half_as_ushort(y_two[k]) ? 1 : 0;
    EXPECT_EQ(yd, 0u) << "the commit-only pass changed the main pass's y rows";

    double ss = 0.0;
    for (float x : want)
        ss += double(x) * double(x);
    const double scale = std::max(1e-9, std::sqrt(ss / static_cast<double>(want.size())));
    auto worst = [&](size_t off) {
        double w = 0.0;
        for (size_t k = off; k < off + state_elems; k++)
            w = std::max(w, std::abs(double(want[k]) - double(got[k])));
        return w;
    };
    for (int sl = 0; sl < n_slots; sl++)
        EXPECT_LT(worst(static_cast<size_t>(sl) * state_elems) / scale, 1e-3) << "slot " << sl;
    // The three committed states differ pairwise (every row moved the state).
    for (int i = 0; i < s.n_seq; i++) {
        const int ab[3][2] = {{live[i], spare2[i]}, {spare2[i], spare[i]}, {live[i], spare[i]}};
        for (const auto& p : ab) {
            double moved = 0.0;
            for (size_t k = 0; k < state_elems; k++)
                moved = std::max(moved, std::abs(double(got[static_cast<size_t>(p[0]) * state_elems + k]) -
                                                 double(got[static_cast<size_t>(p[1]) * state_elems + k])));
            EXPECT_GT(moved / scale, 1e-3)
                << "group " << i << ": slots " << p[0] << " and " << p[1] << " match";
        }
    }

    cudaFree(d_conv);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_Alog);
    cudaFree(d_dtb);
    cudaFree(d_pool);
    cudaFree(d_y);
    cudaFree(d_live);
    cudaFree(d_spare);
    cudaFree(d_snaps);
    cudaFree(d_lens);
}

// The conv half: windows at 3 rows -> spare, 2 rows -> spare2, 1 row in place, bit-exact.
TEST_F(GdnVerifyTwoDraftsTest, ConvCommitsOneTwoAndThreeRowWindows) {
    const int n_seq = 4, n_tokens = 3, channels = 2048, ksize = 4, n_slots = 12;
    const std::vector<int> live{5, 1, 3, 0};
    const std::vector<int> spare{2, 7, 4, 6};
    const std::vector<int> spare2{9, 11, 8, 10};
    std::vector<int> snaps(live);
    snaps.insert(snaps.end(), spare2.begin(), spare2.end());
    const size_t win = static_cast<size_t>(channels) * ksize;
    const size_t pool_elems = static_cast<size_t>(n_slots) * win;
    const size_t rows = static_cast<size_t>(n_seq) * n_tokens;

    std::vector<float> h_pool_init(pool_elems), h_w(win), h_b(channels), h_xf(rows * channels);
    fill(h_pool_init, 113, -1.0f, 1.0f);
    fill(h_w, 224, -0.5f, 0.5f);
    fill(h_b, 335, -0.2f, 0.2f);
    fill(h_xf, 446, -2.0f, 2.0f);
    std::vector<half> h_wh(win), h_bh(channels), h_x(h_xf.size());
    for (size_t i = 0; i < win; i++)
        h_wh[i] = __float2half(h_w[i]);
    for (int i = 0; i < channels; i++)
        h_bh[i] = __float2half(h_b[i]);
    for (size_t i = 0; i < h_xf.size(); i++)
        h_x[i] = __float2half(h_xf[i]);

    float *d_pool = nullptr, *d_out = nullptr;
    half *d_w = nullptr, *d_b = nullptr, *d_x = nullptr;
    int *d_live = nullptr, *d_spare = nullptr, *d_snaps = nullptr, *d_lens = nullptr;
    ASSERT_EQ(cudaMalloc(&d_pool, pool_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, rows * channels * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, win * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, channels * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, h_x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_live, n_seq * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_spare, n_seq * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_snaps, snaps.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_lens, 3 * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_w, h_wh.data(), win * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_bh.data(), channels * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_live, live.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spare, spare.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_snaps, snaps.data(), snaps.size() * sizeof(int), cudaMemcpyHostToDevice);
    const int lens[3] = {3, 1, 2};
    cudaMemcpy(d_lens, lens, sizeof(lens), cudaMemcpyHostToDevice);

    Tensor w{};
    w.data = d_w;
    w.ndim = 2;
    w.shape[0] = channels;
    w.shape[1] = ksize;
    w.qtype = QType::F16;
    Tensor b{};
    b.data = d_b;
    b.ndim = 1;
    b.shape[0] = channels;
    b.qtype = QType::F16;

    std::vector<float> want(pool_elems), out_a(rows * channels);
    std::copy(h_pool_init.begin(), h_pool_init.end(), want.begin());
    for (int i = 0; i < n_seq; i++) {
        for (int n = 1; n <= 3; n++) {
            cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
            Tensor xs{};
            xs.data = d_x + static_cast<size_t>(i) * n_tokens * channels;
            xs.ndim = 2;
            xs.shape[0] = n;
            xs.shape[1] = channels;
            xs.qtype = QType::F16;
            ssm_conv1d_prefill_f32_silu(d_pool + static_cast<size_t>(live[i]) * win, xs, w, b,
                                        d_out + static_cast<size_t>(i) * n_tokens * channels, ksize, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const int dst = n == 1 ? live[i] : (n == 2 ? spare2[i] : spare[i]);
            cudaMemcpy(want.data() + static_cast<size_t>(dst) * win,
                       d_pool + static_cast<size_t>(live[i]) * win, win * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
    }
    cudaMemcpy(out_a.data(), d_out, out_a.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, rows * channels * sizeof(float));
    Tensor xg{};
    xg.data = d_x;
    xg.ndim = 2;
    xg.shape[0] = static_cast<int64_t>(rows);
    xg.shape[1] = channels;
    xg.qtype = QType::F16;
    ssm_conv1d_prefill_f32_silu_grouped(d_pool, d_live, static_cast<int64_t>(win), n_seq, xg, w, b, d_out,
                                        ksize, nullptr, d_lens, /*conv_snap=*/nullptr, d_lens + 1,
                                        /*conv_prev=*/nullptr,
                                        /*out_slots=*/d_spare, /*snap_slots=*/d_snaps, /*snap_rows=*/2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> got(pool_elems), out_b(rows * channels);
    cudaMemcpy(got.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b.data(), d_out, out_b.size() * sizeof(float), cudaMemcpyDeviceToHost);

    size_t wd = 0, od = 0;
    for (size_t i = 0; i < pool_elems; i++)
        if (want[i] != got[i])
            wd++;
    for (size_t i = 0; i < out_a.size(); i++)
        if (out_a[i] != out_b[i])
            od++;
    EXPECT_EQ(wd, 0u) << "conv windows differ in " << wd << " of " << pool_elems;
    EXPECT_EQ(od, 0u) << "conv outputs differ in " << od << " elements";
    size_t moved = 0;
    for (int i = 0; i < n_seq; i++)
        for (size_t k = 0; k < win; k++)
            if (got[static_cast<size_t>(spare2[i]) * win + k] != got[static_cast<size_t>(live[i]) * win + k])
                moved++;
    EXPECT_GT(moved, 0u) << "spare2 and live windows are identical";

    cudaFree(d_pool);
    cudaFree(d_out);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_live);
    cudaFree(d_spare);
    cudaFree(d_snaps);
    cudaFree(d_lens);
}

// ---- factor row: one spare per group, draft 1 also carried as a factor row / conv tap ------------------

float state_to_float(float x) { return x; }
float state_to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename StateT>
StateT state_from_float(float x) {
    if constexpr (std::is_same_v<StateT, float>)
        return x;
    else
        return __float2bfloat16(x);
}

template <typename StateT>
void scan_single(const float* conv, int conv_channels, const half* alpha, const half* beta,
                 const float* A_log, const float* dt_bias, StateT* h, half* y, int n, const ScanShape& s) {
    if constexpr (std::is_same_v<StateT, float>)
        gdn_scan_fused_f32(conv, conv_channels, alpha, beta, A_log, dt_bias, h, y, n, s.n_heads, s.head_dim,
                           s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1);
    else
        gdn_scan_fused_bf16(conv, conv_channels, alpha, beta, A_log, dt_bias, h, y, n, s.n_heads, s.head_dim,
                            s.state_size, s.n_groups, nullptr, /*grouped_layout=*/1);
}

template <typename StateT>
void scan_batched(const float* conv, int conv_channels, const half* alpha, const half* beta,
                  const float* A_log, const float* dt_bias, StateT* pool, const int* seq_slots,
                  int64_t slot_stride, half* y, const ScanShape& s, const int* rows_n, const int* snap_n,
                  const int* outs, const int* snaps, float* fac_out, int fac_stride) {
    if constexpr (std::is_same_v<StateT, float>)
        gdn_scan_fused_f32_batched(conv, conv_channels, alpha, beta, A_log, dt_bias, pool, seq_slots,
                                   slot_stride, y, s.n_seq, s.n_tokens, s.n_heads, s.head_dim, s.state_size,
                                   s.n_groups, nullptr, /*grouped_layout=*/1, rows_n,
                                   /*seq_row_offsets=*/nullptr, /*h_snap=*/nullptr, snap_n, outs, snaps,
                                   fac_out, /*fac_in=*/nullptr, fac_stride);
    else
        gdn_scan_fused_bf16_batched(conv, conv_channels, alpha, beta, A_log, dt_bias, pool, seq_slots,
                                    slot_stride, y, s.n_seq, s.n_tokens, s.n_heads, s.head_dim, s.state_size,
                                    s.n_groups, nullptr, /*grouped_layout=*/1, rows_n,
                                    /*seq_row_offsets=*/nullptr, /*h_snap=*/nullptr, snap_n, outs, snaps,
                                    fac_out, /*fac_in=*/nullptr, fac_stride);
}

// As run_gdn stages the factor rows at D drafts: ONE pass (1 row in place into L_z, D + 1 rows into the spare
// P_z, rows 2 .. D as factor rows [z * (D - 1) + level]), then gdn_factor_fold(dst L_z, rows from z * (D -
// 1), count c_z) = the (1 + c_z)-row state in the live slot. c_z cycles 1 .. D - 1 over the groups. n_seq 4 =
// 32 CTAs (column split), n_seq 13 = 104 CTAs (no column split, limit 96).
template <typename StateT>
void run_scan_factor_row(int n_seq, int drafts) {
    const bool bf16 = !std::is_same_v<StateT, float>;
    const int T = drafts + 1, L = drafts - 1;
    const ScanShape s{n_seq, /*n_tokens=*/T};
    const int n_slots = (2 * n_seq) + 4;  // 4 slots in no table: must stay untouched
    std::vector<int> perm(n_slots);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::mt19937(4177));
    const std::vector<int> live(perm.begin(), perm.begin() + n_seq);
    const std::vector<int> spare(perm.begin() + n_seq, perm.begin() + (2 * n_seq));
    const int conv_channels = 2 * s.n_groups * s.state_size + s.n_heads * s.head_dim;
    const int inner = s.n_heads * s.head_dim;
    const size_t rows = static_cast<size_t>(s.n_seq) * s.n_tokens;
    const size_t state_elems = static_cast<size_t>(s.n_heads) * s.state_size * s.head_dim;
    const size_t pool_elems = static_cast<size_t>(n_slots) * state_elems;
    const size_t pool_bytes = pool_elems * sizeof(StateT);
    const int fac_stride = gdn_factor_floats_per_head(s.state_size, s.head_dim);
    // Rows are indexed by group and level; one more row than the scan may write, which must stay zero.
    const size_t fac_row = static_cast<size_t>(s.n_heads) * fac_stride;
    const size_t fac_elems = ((static_cast<size_t>(n_seq) * L) + 1) * fac_row;
    std::vector<int> count(n_seq), row0(n_seq);
    for (int i = 0; i < n_seq; i++) {
        count[i] = 1 + (i % L);
        row0[i] = i * L;
    }

    std::vector<float> h_conv(rows * conv_channels), h_alpha_f(rows * s.n_heads), h_beta_f(rows * s.n_heads);
    std::vector<float> h_Alog(s.n_heads), h_dtb(s.n_heads), h_init_f(pool_elems);
    fill(h_conv, 6091);
    fill(h_alpha_f, 2713, -2.0f, 2.0f);
    fill(h_beta_f, 8527, -2.0f, 2.0f);
    fill(h_Alog, 1429, -4.0f, -0.5f);
    fill(h_dtb, 3301, -1.0f, 1.0f);
    fill(h_init_f, 7753, -0.5f, 0.5f);
    std::vector<half> h_alpha(h_alpha_f.size()), h_beta(h_beta_f.size());
    for (size_t i = 0; i < h_alpha_f.size(); i++)
        h_alpha[i] = __float2half(h_alpha_f[i]);
    for (size_t i = 0; i < h_beta_f.size(); i++)
        h_beta[i] = __float2half(h_beta_f[i]);
    std::vector<StateT> h_init(pool_elems);
    for (size_t i = 0; i < pool_elems; i++)
        h_init[i] = state_from_float<StateT>(h_init_f[i]);

    float *d_conv = nullptr, *d_Alog = nullptr, *d_dtb = nullptr, *d_fac = nullptr;
    StateT* d_pool = nullptr;
    half *d_alpha = nullptr, *d_beta = nullptr, *d_y = nullptr;
    int *d_live = nullptr, *d_spare = nullptr, *d_lens = nullptr, *d_fold = nullptr;
    ASSERT_EQ(cudaMalloc(&d_conv, h_conv.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_alpha, h_alpha.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_beta, h_beta.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_Alog, h_Alog.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dtb, h_dtb.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_pool, pool_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_fac, fac_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, rows * inner * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_live, live.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_spare, spare.size() * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_lens, T * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_fold, 2 * n_seq * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_fold, row0.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fold + n_seq, count.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv, h_conv.data(), h_conv.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, h_alpha.data(), h_alpha.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta.data(), h_beta.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Alog, h_Alog.data(), h_Alog.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dtb, h_dtb.data(), h_dtb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_live, live.data(), live.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spare, spare.data(), spare.size() * sizeof(int), cudaMemcpyHostToDevice);
    std::vector<int> lens(T);
    std::iota(lens.begin(), lens.end(), 1);
    cudaMemcpy(d_lens, lens.data(), T * sizeof(int), cudaMemcpyHostToDevice);
    const int *d_one = d_lens, *d_all = d_lens + (T - 1);

    auto seq_ptr = [&](int i, size_t per_row) { return static_cast<size_t>(i) * s.n_tokens * per_row; };
    auto reset_pool = [&] { cudaMemcpy(d_pool, h_init.data(), pool_bytes, cudaMemcpyHostToDevice); };
    auto read_pool = [&](std::vector<float>& out) {
        std::vector<StateT> raw(pool_elems);
        cudaMemcpy(raw.data(), d_pool, pool_bytes, cudaMemcpyDeviceToHost);
        out.resize(pool_elems);
        for (size_t k = 0; k < pool_elems; k++)
            out[k] = state_to_float(raw[k]);
    };
    std::vector<float> init;
    reset_pool();
    read_pool(init);

    // ---- reference A (single-sequence launcher): n rows from L_z. want_pre: 1 row at L_z, T rows at P_z;
    // want_post: 1 + c_z rows at L_z, T rows at P_z.
    std::vector<float> want_pre(init), want_post(init);
    std::vector<StateT> one(state_elems);
    for (int i = 0; i < s.n_seq; i++) {
        StateT* h_live = d_pool + static_cast<size_t>(live[i]) * state_elems;
        const StateT* h_live_init = h_init.data() + static_cast<size_t>(live[i]) * state_elems;
        for (int n = 1; n <= T; n++) {
            cudaMemcpy(h_live, h_live_init, state_elems * sizeof(StateT), cudaMemcpyHostToDevice);
            scan_single<StateT>(d_conv + seq_ptr(i, conv_channels), conv_channels,
                                d_alpha + seq_ptr(i, s.n_heads), d_beta + seq_ptr(i, s.n_heads), d_Alog,
                                d_dtb, h_live, d_y + seq_ptr(i, inner), n, s);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            cudaMemcpy(one.data(), h_live, state_elems * sizeof(StateT), cudaMemcpyDeviceToHost);
            const size_t dst = static_cast<size_t>(n == T ? spare[i] : live[i]) * state_elems;
            for (size_t k = 0; k < state_elems; k++) {
                const float v = state_to_float(one[k]);
                if (n == T)
                    want_pre[dst + k] = want_post[dst + k] = v;
                if (n == 1)
                    want_pre[dst + k] = v;
                if (n == 1 + count[i])
                    want_post[dst + k] = v;
            }
        }
    }

    // ---- reference B (same batched kernel, no factor row): T rows -> P_z, 1 row in place. The factor-row
    // launch must reproduce its pool and y rows bit for bit.
    std::vector<float> full, two(pool_elems), part;
    std::vector<half> y_full(rows * inner), y_row(rows * inner);
    reset_pool();
    scan_batched<StateT>(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb, d_pool, d_live,
                         static_cast<int64_t>(state_elems), d_y, s, d_all, d_one, d_spare, d_live, nullptr,
                         0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    read_pool(full);
    cudaMemcpy(y_full.data(), d_y, y_full.size() * sizeof(half), cudaMemcpyDeviceToHost);
    // ---- reference C (same batched kernel): 1 + c rows in place, the state a fold of c rows must land on.
    for (int c = 1; c <= L; c++) {
        reset_pool();
        scan_batched<StateT>(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb, d_pool, d_live,
                             static_cast<int64_t>(state_elems), d_y, s, d_lens + c, /*snap_n=*/nullptr,
                             /*outs=*/nullptr, /*snaps=*/nullptr, /*fac_out=*/nullptr, 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        read_pool(part);
        for (int i = 0; i < s.n_seq; i++)
            if (count[i] == c)
                std::copy_n(part.begin() + static_cast<size_t>(live[i]) * state_elems, state_elems,
                            two.begin() + static_cast<size_t>(live[i]) * state_elems);
    }

    // ---- the factor rows
    reset_pool();
    cudaMemset(d_y, 0, rows * inner * sizeof(half));
    cudaMemset(d_fac, 0, fac_elems * sizeof(float));
    scan_batched<StateT>(d_conv, conv_channels, d_alpha, d_beta, d_Alog, d_dtb, d_pool, d_live,
                         static_cast<int64_t>(state_elems), d_y, s, d_all, d_one, d_spare, d_live, d_fac,
                         fac_stride);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> pre, post, h_fac(fac_elems);
    read_pool(pre);
    cudaMemcpy(y_row.data(), d_y, y_row.size() * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fac.data(), d_fac, fac_elems * sizeof(float), cudaMemcpyDeviceToHost);
    // 2 drafts: the default count (nullptr = 1 row per entry).
    gdn_factor_fold(d_pool, bf16, static_cast<int64_t>(state_elems * sizeof(StateT)),
                    /*layer_stride_bytes=*/0, d_fac, /*fac_layer_stride=*/0, /*n_layers=*/1, d_live, d_fold,
                    s.n_seq, s.n_heads, s.state_size, s.head_dim, fac_stride, nullptr,
                    L == 1 ? nullptr : d_fold + n_seq);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    read_pool(post);

    size_t yd = 0, sd = 0;
    for (size_t k = 0; k < y_full.size(); k++)
        yd += __half_as_ushort(y_full[k]) != __half_as_ushort(y_row[k]) ? 1 : 0;
    for (size_t k = 0; k < pool_elems; k++)
        sd += pre[k] != full[k] ? 1 : 0;
    EXPECT_EQ(yd, 0u) << "the factor row changed the y rows";
    EXPECT_EQ(sd, 0u) << "the factor rows changed the 1-row or all-rows states";

    double ss = 0.0;
    for (float x : want_post)
        ss += double(x) * double(x);
    const double scale = std::max(1e-9, std::sqrt(ss / static_cast<double>(want_post.size())));
    // bf16 keeps 8 significand bits: one rounding is <= 2^-8 |x|, a flipped rounding (1 ulp) <= 2^-7 |x|.
    const double half_ulp = bf16 ? 1.05 / 256.0 : 0.0;
    const double ulp = 2.0 * half_ulp;
    std::vector<bool> is_live(n_slots, false), is_spare(n_slots, false);
    for (int i = 0; i < s.n_seq; i++) {
        is_live[live[i]] = true;
        is_spare[spare[i]] = true;
    }
    for (int sl = 0; sl < n_slots; sl++) {
        const size_t off = static_cast<size_t>(sl) * state_elems;
        if (!is_live[sl] && !is_spare[sl]) {
            size_t touched = 0;
            for (size_t k = off; k < off + state_elems; k++)
                touched += (pre[k] != init[k] || post[k] != init[k]) ? 1 : 0;
            EXPECT_EQ(touched, 0u) << "slot " << sl << " is in no table";
            continue;
        }
        // Before the fold: L_z = 1 row, P_z = T rows (the existing test's 1e-3 of the state rms, plus
        // one flipped bf16 rounding between the two launchers).
        size_t bad_pre = 0, bad_ref = 0, bad_two = 0, spare_moved = 0, folded = 0;
        for (size_t k = off; k < off + state_elems; k++) {
            const double launcher_tol = (1e-3 * scale) + (ulp * std::abs(double(want_post[k])));
            bad_pre += std::abs(double(pre[k]) - double(want_pre[k])) >
                               (1e-3 * scale) + (ulp * std::abs(double(want_pre[k])))
                           ? 1
                           : 0;
            if (is_spare[sl]) {
                spare_moved += post[k] != pre[k] ? 1 : 0;
                continue;
            }
            // After the fold, against the same kernel's 1 + c rows in place. f32: same expression, 1e-5 of
            // the rms covers FMA contraction. bf16: the 1-row state was rounded once before the fold (<= g *
            // 2^-8 |S1|, g <= 1) and the result once more (<= 1 ulp of the reference apart).
            const double fold_tol = (1e-5 * scale) + (half_ulp * std::abs(double(pre[k]))) +
                                    (ulp * std::abs(double(two[k])));
            const double err = std::abs(double(post[k]) - double(two[k]));
            bad_two += err > fold_tol ? 1 : 0;
            bad_ref += std::abs(double(post[k]) - double(want_post[k])) >
                               launcher_tol + (half_ulp * std::abs(double(pre[k])))
                           ? 1
                           : 0;
            // The fold must be visible past its own tolerance, or the check above proves nothing.
            folded += std::abs(double(post[k]) - double(pre[k])) > 4.0 * fold_tol ? 1 : 0;
        }
        EXPECT_EQ(bad_pre, 0u) << "slot " << sl << " before the fold";
        if (is_spare[sl]) {
            EXPECT_EQ(spare_moved, 0u) << "the fold wrote spare slot " << sl;
            continue;
        }
        EXPECT_EQ(bad_two, 0u) << "live " << sl << " vs the same kernel's state after the folded rows";
        EXPECT_EQ(bad_ref, 0u) << "live " << sl << " vs the single-sequence state after the folded rows";
        EXPECT_GT(folded, state_elems / 100) << "live " << sl << ": the fold moved (almost) nothing";
    }
    // The pass must leave a row (g != 0) for every group, level and head, and nothing past them.
    for (int r = 0; r < n_seq * L; r++)
        for (int h = 0; h < s.n_heads; h++)
            EXPECT_NE(h_fac[(static_cast<size_t>(r) * fac_row) + (static_cast<size_t>(h) * fac_stride)], 0.0f)
                << "no factor row at group " << (r / L) << " level " << (r % L) << " head " << h;
    size_t past = 0;
    for (size_t k = static_cast<size_t>(n_seq) * L * fac_row; k < fac_elems; k++)
        past += h_fac[k] != 0.0f ? 1 : 0;
    EXPECT_EQ(past, 0u) << "the scan wrote past the last group's rows";

    cudaFree(d_conv);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_Alog);
    cudaFree(d_dtb);
    cudaFree(d_pool);
    cudaFree(d_fac);
    cudaFree(d_y);
    cudaFree(d_live);
    cudaFree(d_spare);
    cudaFree(d_lens);
    cudaFree(d_fold);
}

TEST_F(GdnVerifyTwoDraftsTest, ScanFactorRowFoldsTheLiveSlotToTheTwoRowState) {
    for (int n_seq : {4, 13}) {
        SCOPED_TRACE(testing::Message() << "f32 n_seq " << n_seq);
        run_scan_factor_row<float>(n_seq, /*drafts=*/2);
    }
}

TEST_F(GdnVerifyTwoDraftsTest, ScanFactorRowFoldsTheLiveSlotToTheTwoRowStateBf16) {
    for (int n_seq : {4, 13}) {
        SCOPED_TRACE(testing::Message() << "bf16 n_seq " << n_seq);
        run_scan_factor_row<__nv_bfloat16>(n_seq, /*drafts=*/2);
    }
}

// 3 drafts: 2 factor rows per group, the groups fold 1 or 2 of them in turn.
TEST_F(GdnVerifyTwoDraftsTest, ScanFactorRowsFoldOneOrTwoDraftsOfThree) {
    for (int n_seq : {4, 13}) {
        SCOPED_TRACE(testing::Message() << "f32 n_seq " << n_seq);
        run_scan_factor_row<float>(n_seq, /*drafts=*/3);
    }
}

TEST_F(GdnVerifyTwoDraftsTest, ScanFactorRowsFoldOneOrTwoDraftsOfThreeBf16) {
    for (int n_seq : {4, 13}) {
        SCOPED_TRACE(testing::Message() << "bf16 n_seq " << n_seq);
        run_scan_factor_row<__nv_bfloat16>(n_seq, /*drafts=*/3);
    }
}

// 4 drafts: 3 factor rows per group, the groups fold 1, 2 or 3 of them in turn.
TEST_F(GdnVerifyTwoDraftsTest, ScanFactorRowsFoldOneToThreeDraftsOfFour) {
    for (int n_seq : {4, 13}) {
        SCOPED_TRACE(testing::Message() << "f32 n_seq " << n_seq);
        run_scan_factor_row<float>(n_seq, /*drafts=*/4);
    }
}

TEST_F(GdnVerifyTwoDraftsTest, ScanFactorRowsFoldOneToThreeDraftsOfFourBf16) {
    for (int n_seq : {4, 13}) {
        SCOPED_TRACE(testing::Message() << "bf16 n_seq " << n_seq);
        run_scan_factor_row<__nv_bfloat16>(n_seq, /*drafts=*/4);
    }
}

// 8 drafts (kMtpBoundVerifyMaxDepth): 7 factor rows per group = kGdnFactorFoldMaxRows, 13 groups fold 1 to 7.
TEST_F(GdnVerifyTwoDraftsTest, ScanFactorRowsFoldOneToSevenDraftsOfEight) {
    for (int n_seq : {4, 13}) {
        SCOPED_TRACE(testing::Message() << "f32 n_seq " << n_seq);
        run_scan_factor_row<float>(n_seq, /*drafts=*/8);
    }
}

TEST_F(GdnVerifyTwoDraftsTest, ScanFactorRowsFoldOneToSevenDraftsOfEightBf16) {
    for (int n_seq : {4, 13}) {
        SCOPED_TRACE(testing::Message() << "bf16 n_seq " << n_seq);
        run_scan_factor_row<__nv_bfloat16>(n_seq, /*drafts=*/8);
    }
}

// The conv half at D drafts: D + 1 rows -> spare, 1 row in place, rows 2 .. D as taps [z * (D - 1) + level],
// then ssm_conv_tap_fold(window live, taps from z * (D - 1), count c_z) gives the (1 + c_z)-row window in
// the live slot, bit-exact. c_z cycles D - 1 .. 1 over the groups.
void run_conv_factor_row(int drafts) {
    const int n_seq = 4, n_tokens = drafts + 1, L = drafts - 1, channels = 2048, ksize = 4, n_slots = 12;
    const std::vector<int> live{5, 1, 3, 0};
    const std::vector<int> spare{2, 7, 4, 6};
    const size_t win = static_cast<size_t>(channels) * ksize;
    const size_t pool_elems = static_cast<size_t>(n_slots) * win;
    const size_t rows = static_cast<size_t>(n_seq) * n_tokens;
    // One more tap row than the stash may write, which must stay zero.
    const size_t tap_elems = ((static_cast<size_t>(n_seq) * L) + 1) * channels;
    std::vector<int> fold(2 * n_seq);  // [first tap row x n_seq][count x n_seq]
    for (int i = 0; i < n_seq; i++) {
        fold[i] = i * L;
        fold[n_seq + i] = L - (i % L);
    }
    const int* count = fold.data() + n_seq;

    std::vector<float> h_pool_init(pool_elems), h_w(win), h_b(channels), h_xf(rows * channels);
    fill(h_pool_init, 557, -1.0f, 1.0f);
    fill(h_w, 668, -0.5f, 0.5f);
    fill(h_b, 779, -0.2f, 0.2f);
    fill(h_xf, 881, -2.0f, 2.0f);
    std::vector<half> h_wh(win), h_bh(channels), h_x(h_xf.size());
    for (size_t i = 0; i < win; i++)
        h_wh[i] = __float2half(h_w[i]);
    for (int i = 0; i < channels; i++)
        h_bh[i] = __float2half(h_b[i]);
    for (size_t i = 0; i < h_xf.size(); i++)
        h_x[i] = __float2half(h_xf[i]);

    float *d_pool = nullptr, *d_out = nullptr;
    half *d_w = nullptr, *d_b = nullptr, *d_x = nullptr, *d_tap = nullptr;
    int *d_live = nullptr, *d_spare = nullptr, *d_lens = nullptr, *d_fold = nullptr;
    ASSERT_EQ(cudaMalloc(&d_pool, pool_elems * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, rows * channels * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, win * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_b, channels * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_x, h_x.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_tap, tap_elems * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_live, n_seq * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_spare, n_seq * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_lens, n_tokens * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_fold, fold.size() * sizeof(int)), cudaSuccess);
    cudaMemcpy(d_w, h_wh.data(), win * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_bh.data(), channels * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_live, live.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spare, spare.data(), n_seq * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fold, fold.data(), fold.size() * sizeof(int), cudaMemcpyHostToDevice);
    std::vector<int> lens(n_tokens);
    std::iota(lens.begin(), lens.end(), 1);
    cudaMemcpy(d_lens, lens.data(), n_tokens * sizeof(int), cudaMemcpyHostToDevice);
    const int *d_one = d_lens, *d_all = d_lens + (n_tokens - 1);

    Tensor w{};
    w.data = d_w;
    w.ndim = 2;
    w.shape[0] = channels;
    w.shape[1] = ksize;
    w.qtype = QType::F16;
    Tensor b{};
    b.data = d_b;
    b.ndim = 1;
    b.shape[0] = channels;
    b.qtype = QType::F16;

    // Reference: n rows from L_z. want_pre: 1 row at L_z, all rows at P_z; want_post: 1 + c_z rows at L_z.
    std::vector<float> want_pre(h_pool_init), want_post(h_pool_init), out_a(rows * channels);
    for (int i = 0; i < n_seq; i++) {
        for (int n = 1; n <= n_tokens; n++) {
            cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
            Tensor xs{};
            xs.data = d_x + static_cast<size_t>(i) * n_tokens * channels;
            xs.ndim = 2;
            xs.shape[0] = n;
            xs.shape[1] = channels;
            xs.qtype = QType::F16;
            ssm_conv1d_prefill_f32_silu(d_pool + static_cast<size_t>(live[i]) * win, xs, w, b,
                                        d_out + static_cast<size_t>(i) * n_tokens * channels, ksize, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const float* src = d_pool + static_cast<size_t>(live[i]) * win;
            const size_t dst = static_cast<size_t>(n == n_tokens ? spare[i] : live[i]) * win;
            if (n == 1 || n == n_tokens)
                cudaMemcpy(want_pre.data() + dst, src, win * sizeof(float), cudaMemcpyDeviceToHost);
            if (n == 1 + count[i] || n == n_tokens)
                cudaMemcpy(want_post.data() + dst, src, win * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    cudaMemcpy(out_a.data(), d_out, out_a.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // As run_gdn stages it: commit at all rows into the spare, at 1 row in place, taps = rows 2 .. D.
    cudaMemcpy(d_pool, h_pool_init.data(), pool_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, rows * channels * sizeof(float));
    cudaMemset(d_tap, 0, tap_elems * sizeof(half));
    Tensor xg{};
    xg.data = d_x;
    xg.ndim = 2;
    xg.shape[0] = static_cast<int64_t>(rows);
    xg.shape[1] = channels;
    xg.qtype = QType::F16;
    ssm_conv1d_prefill_f32_silu_grouped(d_pool, d_live, static_cast<int64_t>(win), n_seq, xg, w, b, d_out,
                                        ksize, nullptr, d_all, /*conv_snap=*/nullptr, d_one,
                                        /*conv_prev=*/nullptr,
                                        /*out_slots=*/d_spare, /*snap_slots=*/d_live, /*snap_rows=*/1);
    ssm_conv_tap_stash_mid(d_tap, xg.data, n_tokens, channels, d_one, L, n_seq, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<float> pre(pool_elems), post(pool_elems), out_b(rows * channels);
    std::vector<half> h_tap(tap_elems);
    cudaMemcpy(pre.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b.data(), d_out, out_b.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tap.data(), d_tap, tap_elems * sizeof(half), cudaMemcpyDeviceToHost);
    ssm_conv_tap_fold(d_pool, static_cast<int64_t>(win * sizeof(float)), /*layer_stride_bytes=*/0, d_tap,
                      /*tap_layer_stride=*/0, /*n_layers=*/1, channels, ksize, d_live, d_fold, n_seq, nullptr,
                      L == 1 ? nullptr : d_fold + n_seq);  // 2 drafts: the default count of 1
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cudaMemcpy(post.data(), d_pool, pool_elems * sizeof(float), cudaMemcpyDeviceToHost);

    size_t pd = 0, wd = 0, od = 0, td = 0;
    for (size_t i = 0; i < pool_elems; i++) {
        pd += pre[i] != want_pre[i] ? 1 : 0;
        wd += post[i] != want_post[i] ? 1 : 0;
    }
    for (size_t i = 0; i < out_a.size(); i++)
        od += out_a[i] != out_b[i] ? 1 : 0;
    // Tap [z * L + level] is group z's input row 1 + level; the row past the last group stays zero.
    std::vector<half> want_tap(tap_elems, __float2half(0.0f));
    for (int i = 0; i < n_seq * L; i++) {
        const size_t src_row = ((static_cast<size_t>(i / L) * n_tokens) + 1 + (i % L)) * channels;
        for (int c = 0; c < channels; c++)
            want_tap[(static_cast<size_t>(i) * channels) + c] = h_x[src_row + c];
    }
    for (size_t i = 0; i < tap_elems; i++)
        td += __half_as_ushort(h_tap[i]) != __half_as_ushort(want_tap[i]) ? 1 : 0;
    EXPECT_EQ(pd, 0u) << "windows before the fold differ in " << pd << " of " << pool_elems;
    EXPECT_EQ(wd, 0u) << "windows after the fold differ in " << wd << " of " << pool_elems;
    EXPECT_EQ(od, 0u) << "conv outputs differ in " << od << " elements";
    EXPECT_EQ(td, 0u) << "taps differ in " << td << " of " << tap_elems;
    size_t moved = 0;
    for (int i = 0; i < n_seq; i++) {
        const size_t off = static_cast<size_t>(live[i]) * win;
        for (size_t k = 0; k < win; k++)
            moved += post[off + k] != pre[off + k] ? 1 : 0;
    }
    EXPECT_GT(moved, 0u) << "the fold left the live windows as they were";

    cudaFree(d_pool);
    cudaFree(d_out);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_tap);
    cudaFree(d_live);
    cudaFree(d_spare);
    cudaFree(d_lens);
    cudaFree(d_fold);
}

TEST_F(GdnVerifyTwoDraftsTest, ConvFactorRowFoldsTheLiveWindowToTwoRows) {
    run_conv_factor_row(/*drafts=*/2);
}

// 3 drafts: 2 taps per group, the groups fold 1 or 2 of them in turn.
TEST_F(GdnVerifyTwoDraftsTest, ConvFactorRowsFoldOneOrTwoDraftsOfThree) { run_conv_factor_row(/*drafts=*/3); }

// 4 drafts: 3 taps per group, the groups fold 1, 2 or 3 of them in turn.
TEST_F(GdnVerifyTwoDraftsTest, ConvFactorRowsFoldOneToThreeDraftsOfFour) {
    run_conv_factor_row(/*drafts=*/4);
}

// 8 drafts: 7 taps per group, the groups fold 7, 6, 5 and 4 of them (more taps than the 4-row window holds).
TEST_F(GdnVerifyTwoDraftsTest, ConvFactorRowsFoldFourToSevenDraftsOfEight) {
    run_conv_factor_row(/*drafts=*/8);
}

}  // namespace
}  // namespace imp
