// Ragged GDN forward with mixed-step riders (exec/gdn_ragged.h, roadmap row 42): two prompt
// members (300 rows: chunk-parallel scan; 40 rows: fused scan) plus six one-row riders on
// scattered, unordered slots. Rider rows must be bit-identical to a separate batched decode step
// and prompt rows to solo per-sequence runs; a wrong slot or row offset moves whole rows. The
// old one-launch ragged route is the rounding cross-check (rider math is the same recurrence).

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "compute/gdn.h"
#include "compute/ssm.h"
#include "exec/gdn_ragged.h"
#include "memory/ssm_state.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace imp {
namespace {

constexpr int kH = 8, kD = 128, kS = 128, kG = 8, kK = 4;
constexpr int kC = (2 * kG * kS) + (kH * kD);
constexpr int kInner = kH * kD;
constexpr int kSlots = 24;

std::vector<float> rnd(size_t n, uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v)
        x = d(rng);
    return v;
}

std::vector<half> to_half(const std::vector<float>& v) {
    std::vector<half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        h[i] = __float2half(v[i]);
    return h;
}

template <typename T>
T* upload(const std::vector<T>& v) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, std::max<size_t>(v.size(), 1) * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> download(const T* d, size_t n) {
    std::vector<T> v(n);
    EXPECT_EQ(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return v;
}

class GdnRaggedRidersTest : public ::testing::Test {
protected:
    void SetUp() override {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0)
            GTEST_SKIP() << "no CUDA device";
    }
};

// Worst |a-b| over a row range, relative to the reference's RMS (state values can sit near 0).
template <typename T, typename F>
double rel_diff(const std::vector<T>& a, const std::vector<T>& b, size_t lo, size_t hi, F conv) {
    double ss = 0.0, worst = 0.0;
    for (size_t i = lo; i < hi; ++i) {
        const double r = conv(b[i]);
        ss += r * r;
        worst = std::max(worst, std::abs(conv(a[i]) - r));
    }
    return worst / std::max(1e-9, std::sqrt(ss / static_cast<double>(hi - lo)));
}

TEST_F(GdnRaggedRidersTest, RidersMatchSeparateDecodeAndPromptsMatchSolo) {
    // Members: two prompts, then six riders. Slots scattered and unordered (scheduler free list).
    const std::vector<int> rows_of{300, 40, 1, 1, 1, 1, 1, 1};
    const std::vector<int> slots{9, 3, 14, 0, 21, 7, 5, 12};
    const int n_seq = static_cast<int>(rows_of.size()), n_riders = 6, n_pf = n_seq - n_riders;
    std::vector<int> offs(static_cast<size_t>(n_seq) + 1, 0);
    for (int s = 0; s < n_seq; ++s)
        offs[s + 1] = offs[s] + rows_of[s];
    const int n = offs[n_seq];
    const int r0 = offs[n_pf];

    SSMState st;
    ASSERT_TRUE(st.init(/*n_ssm_layers=*/1, kSlots, kC, kK, kH, kD, kS, QType::F32));
    const size_t slab_floats = st.per_seq_bytes() / sizeof(float);
    std::vector<std::vector<float>> init(kSlots);
    for (int s = 0; s < kSlots; ++s)
        init[s] = rnd(slab_floats, 100 + s, -0.5f, 0.5f);
    auto reset_pool = [&]() {
        for (int s = 0; s < kSlots; ++s)
            ASSERT_EQ(cudaMemcpy(st.seq_base(s), init[s].data(), st.per_seq_bytes(), cudaMemcpyHostToDevice),
                      cudaSuccess);
    };
    auto read_pool = [&]() {
        std::vector<float> all;
        for (int s = 0; s < kSlots; ++s) {
            auto v = download(static_cast<const float*>(st.seq_base(s)), slab_floats);
            all.insert(all.end(), v.begin(), v.end());
        }
        return all;
    };

    half* d_x = upload(to_half(rnd(static_cast<size_t>(n) * kC, 1, -2.f, 2.f)));
    half* d_w = upload(to_half(rnd(static_cast<size_t>(kC) * kK, 2, -0.5f, 0.5f)));
    half* d_b = upload(to_half(rnd(kC, 3, -0.2f, 0.2f)));
    half* d_alpha = upload(to_half(rnd(static_cast<size_t>(n) * kH, 4, -2.f, 2.f)));
    half* d_beta = upload(to_half(rnd(static_cast<size_t>(n) * kH, 5, -2.f, 2.f)));
    float* d_alog = upload(rnd(kH, 6, -4.f, -0.5f));
    float* d_dtb = upload(rnd(kH, 7, -1.f, 1.f));
    int* d_slots = upload(slots);
    int* d_offs = upload(offs);
    int* d_rider_slots = upload(std::vector<int>(slots.begin() + n_pf, slots.end()));
    const size_t ws_bytes = gdn_scan_chunkpar_workspace_bytes(kH);
    float* d_ws = nullptr;
    float* d_conv = nullptr;
    half* d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ws, ws_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_conv, static_cast<size_t>(n) * kC * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(n) * kInner * sizeof(half)), cudaSuccess);

    int64_t wshape[2] = {kC, kK};
    int64_t bshape[1] = {kC};
    int64_t xshape[2] = {n, kC};
    Tensor w(d_w, QType::F16, 2, wshape, true);
    Tensor b(d_b, QType::F16, 1, bshape, true);
    Tensor x(d_x, QType::F16, 2, xshape, true);

    GdnRaggedRows g;
    g.n_seq = n_seq;
    g.n_riders = n_riders;
    g.h_seq_offsets = offs.data();
    g.h_ssm_slots = slots.data();
    g.d_ssm_slots = d_slots;
    GdnRaggedScanIo io;
    io.conv_f32 = d_conv;
    io.conv_channels = kC;
    io.alpha = d_alpha;
    io.beta = d_beta;
    io.A_log = d_alog;
    io.dt_bias = d_dtb;
    io.y = d_y;
    io.n_heads = kH;
    io.head_dim = kD;
    io.state_size = kS;
    io.n_groups = kG;
    io.grouped_layout = 1;
    io.chunkpar_ws = d_ws;
    io.chunkpar_bytes = ws_bytes;
    const size_t yn = static_cast<size_t>(n) * kInner, cn = static_cast<size_t>(n) * kC;

    // Arm A: the mixed step (prompts per member, riders batched).
    reset_pool();
    cudaMemset(d_y, 0, yn * sizeof(half));
    gdn_ragged_conv(g, st, 0, x, w, b, d_conv, kK, nullptr);
    ASSERT_TRUE(gdn_ragged_scan_split(g, st, 0, io, nullptr))
        << "one 300-row prompt must take the split route";
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto conv_a = download(d_conv, cn);
    const auto y_a = download(d_y, yn);
    const auto pool_a = read_pool();

    // Arm B: each prompt solo, then the riders as their own decode step (own buffers, row 0 up).
    reset_pool();
    cudaMemset(d_y, 0, yn * sizeof(half));
    cudaMemset(d_conv, 0, cn * sizeof(float));
    for (int s = 0; s < n_pf; ++s) {
        int64_t sshape[2] = {rows_of[s], kC};
        Tensor xs(d_x + (static_cast<size_t>(offs[s]) * kC), QType::F16, 2, sshape, true);
        float* cs = d_conv + (static_cast<size_t>(offs[s]) * kC);
        ssm_conv1d_prefill_f32_silu(st.conv_state(slots[s], 0), xs, w, b, cs, kK, nullptr);
        auto* hs = static_cast<float*>(st.h_state(slots[s], 0));
        half* ys = d_y + (static_cast<size_t>(offs[s]) * kInner);
        const half* as = d_alpha + (static_cast<size_t>(offs[s]) * kH);
        const half* bs = d_beta + (static_cast<size_t>(offs[s]) * kH);
        if (rows_of[s] >= 128)
            gdn_scan_chunkpar_f32(cs, kC, as, bs, d_alog, d_dtb, hs, ys, rows_of[s], kH, kD, kS, kG, nullptr,
                                  1, d_ws, ws_bytes);
        else
            gdn_scan_fused_f32(cs, kC, as, bs, d_alog, d_dtb, hs, ys, rows_of[s], kH, kD, kS, kG, nullptr, 1);
    }
    half* d_rx = nullptr;
    float* d_rconv = nullptr;
    half *d_ra = nullptr, *d_rb = nullptr, *d_ry = nullptr;
    ASSERT_EQ(cudaMalloc(&d_rx, static_cast<size_t>(n_riders) * kC * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_rconv, static_cast<size_t>(n_riders) * kC * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ra, static_cast<size_t>(n_riders) * kH * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_rb, static_cast<size_t>(n_riders) * kH * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ry, static_cast<size_t>(n_riders) * kInner * sizeof(half)), cudaSuccess);
    cudaMemcpy(d_rx, d_x + (static_cast<size_t>(r0) * kC), static_cast<size_t>(n_riders) * kC * sizeof(half),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_ra, d_alpha + (static_cast<size_t>(r0) * kH),
               static_cast<size_t>(n_riders) * kH * sizeof(half), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_rb, d_beta + (static_cast<size_t>(r0) * kH),
               static_cast<size_t>(n_riders) * kH * sizeof(half), cudaMemcpyDeviceToDevice);
    const int64_t slot_stride = static_cast<int64_t>(st.slot_stride_bytes() / sizeof(float));
    ssm_conv1d_decode_f32_silu_batched(st.conv_state(0, 0), d_rider_slots, slot_stride, d_rx, w, b, d_rconv,
                                       n_riders, kC, kK, nullptr);
    gdn_scan_fused_f32_batched(d_rconv, kC, d_ra, d_rb, d_alog, d_dtb, static_cast<float*>(st.h_state(0, 0)),
                               d_rider_slots, slot_stride, d_ry, n_riders, 1, kH, kD, kS, kG, nullptr, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto conv_b = download(d_conv, cn);
    auto y_b = download(d_y, yn);
    {
        const auto rc = download(d_rconv, static_cast<size_t>(n_riders) * kC);
        const auto ry = download(d_ry, static_cast<size_t>(n_riders) * kInner);
        std::copy(rc.begin(), rc.end(), conv_b.begin() + (static_cast<ptrdiff_t>(r0) * kC));
        std::copy(ry.begin(), ry.end(), y_b.begin() + (static_cast<ptrdiff_t>(r0) * kInner));
    }
    const auto pool_b = read_pool();

    auto count_diff = [](const auto& a, const auto& b2, size_t lo, size_t hi) {
        size_t d = 0;
        for (size_t i = lo; i < hi; ++i)
            d += (std::memcmp(&a[i], &b2[i], sizeof(a[i])) != 0) ? 1 : 0;
        return d;
    };
    const size_t pr_c = static_cast<size_t>(r0) * kC, pr_y = static_cast<size_t>(r0) * kInner;
    EXPECT_EQ(count_diff(conv_a, conv_b, 0, pr_c), 0u) << "prompt conv rows differ from the solo runs";
    EXPECT_EQ(count_diff(y_a, y_b, 0, pr_y), 0u) << "prompt scan rows differ from the solo runs";
    EXPECT_EQ(count_diff(conv_a, conv_b, pr_c, cn), 0u) << "rider conv rows differ from the decode step";
    EXPECT_EQ(count_diff(y_a, y_b, pr_y, yn), 0u) << "rider scan rows differ from the decode step";
    EXPECT_EQ(count_diff(pool_a, pool_b, 0, pool_a.size()), 0u) << "recurrent state pool differs";

    // Every touched slot moved, untouched ones did not (a stray write lands in a foreign slot).
    for (int s = 0; s < kSlots; ++s) {
        const size_t lo = static_cast<size_t>(s) * slab_floats;
        size_t moved = 0;
        for (size_t i = 0; i < slab_floats; ++i)
            moved += pool_a[lo + i] != init[s][i] ? 1 : 0;
        const bool used = std::find(slots.begin(), slots.end(), s) != slots.end();
        if (used)
            EXPECT_GT(moved, slab_floats / 4) << "slot " << s << " barely moved: its member did not run";
        else
            EXPECT_EQ(moved, 0u) << "slot " << s << " belongs to no member but was written";
    }

    // Arm C: the old route (per-member prefill conv for riders too, one batched ragged scan).
    // Same recurrence, other kernels: rider rows within rounding of arm A.
    reset_pool();
    GdnRaggedRows g_old = g;
    g_old.n_riders = 0;
    gdn_ragged_conv(g_old, st, 0, x, w, b, d_conv, kK, nullptr);
    gdn_scan_fused_f32_batched(d_conv, kC, d_alpha, d_beta, d_alog, d_dtb,
                               static_cast<float*>(st.h_state(0, 0)), d_slots, slot_stride, d_y, n_seq, 1, kH,
                               kD, kS, kG, nullptr, 1, nullptr, d_offs);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto conv_c = download(d_conv, cn);
    const auto y_c = download(d_y, yn);
    const auto pool_c = read_pool();
    auto f = [](float v) { return static_cast<double>(v); };
    auto h = [](half v) { return static_cast<double>(__half2float(v)); };
    EXPECT_LT(rel_diff(conv_a, conv_c, pr_c, cn, f), 1e-5) << "rider conv: decode vs prefill kernel";
    EXPECT_LT(rel_diff(y_a, y_c, pr_y, yn, h), 1e-2) << "rider scan: decode vs ragged kernel";
    for (int s = n_pf; s < n_seq; ++s) {
        const size_t lo = static_cast<size_t>(slots[s]) * slab_floats;
        EXPECT_LT(rel_diff(pool_a, pool_c, lo, lo + slab_floats, f), 1e-3) << "rider slot " << slots[s];
    }

    for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_w), static_cast<void*>(d_b),
                    static_cast<void*>(d_alpha), static_cast<void*>(d_beta), static_cast<void*>(d_alog),
                    static_cast<void*>(d_dtb), static_cast<void*>(d_slots), static_cast<void*>(d_offs),
                    static_cast<void*>(d_rider_slots), static_cast<void*>(d_ws), static_cast<void*>(d_conv),
                    static_cast<void*>(d_y), static_cast<void*>(d_rx), static_cast<void*>(d_rconv),
                    static_cast<void*>(d_ra), static_cast<void*>(d_rb), static_cast<void*>(d_ry)})
        cudaFree(p);
}

}  // namespace
}  // namespace imp
