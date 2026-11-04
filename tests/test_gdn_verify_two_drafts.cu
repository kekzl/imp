// Bound verify at 2 drafts (speculative.mtp_batch_verify_k 2, docs/plans/2026-09-11-batched-mtp-verify.md):
// groups of 3 rows (last token + 2 drafts). Three states per group: 1 row in place into the live slot L_z,
// 2 rows into spare2 Q_z, 3 rows into spare P_z. The scan writes the 2-row state in a commit-only pass
// before the main pass (run_gdn, ssm_snap_rows 2); the conv takes a third commit launch (snap_rows 2).
// Reference: the single-sequence launchers per group and row count.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "compute/gdn.h"
#include "compute/ssm.h"

#include <algorithm>
#include <cmath>
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

}  // namespace
}  // namespace imp
