// Per-token dynamic FP8 activation scales (quantize_fp8_act_rows + the per-row epilogue of gemm_fp8_smallm
// and gemm_fp8_tile): row 0's output bits must not depend on what the other rows hold. The batch-wide absmax
// it replaced fails the same check (control arm), so the input reaches the defect.

#include "quant/fp8_gemm_smallm.h"
#include "quant/fp8_quant.h"
#include "quant/fp8_row_quant.h"
#include "quant/nvfp4_gemm_tile.h"

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace imp;

namespace {

constexpr int kN = 1024;
constexpr int kK = 4096;
constexpr int kMaxM = 600;  // a prompt chunk on the tile; decode rows use the first 32
constexpr float kWScale = 0.013f;

bool has_device() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::vector<half> random_half(size_t n, uint32_t seed, float amp) {
    std::vector<half> v(n);
    uint32_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h = h * 1664525u + 1013904223u;
        v[i] = __float2half(amp * (static_cast<float>(h >> 8) / 8388608.0f - 1.0f));
    }
    return v;
}

std::vector<uint8_t> random_e4m3(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    uint32_t h = seed;
    for (size_t i = 0; i < n; ++i) {
        h = h * 1664525u + 1013904223u;
        v[i] = static_cast<uint8_t>(
            __nv_fp8_e4m3((static_cast<float>(h >> 8) / 8388608.0f - 1.0f) * 2.0f).__x);
    }
    return v;
}

// Two batches: row 0 identical, rows 1.. differ in content, and batch B carries a 4096x outlier row (row 7),
// which moves any batch-wide absmax and pushes row 0 into the E4M3 subnormals (below 2^-6) there.
struct Batches {
    std::vector<half> a, b;
    Batches() {
        a = random_half(size_t(kMaxM) * kK, 0x51u, 3.0f);
        b = random_half(size_t(kMaxM) * kK, 0xA7u, 3.0f);
        std::copy(a.begin(), a.begin() + kK, b.begin());
        for (int k = 0; k < kK; ++k)
            b[size_t(7) * kK + k] = __float2half(4096.0f * __half2float(b[size_t(7) * kK + k]));
    }
};

struct Dev {
    half* x = nullptr;
    uint8_t* x8 = nullptr;
    uint8_t* w = nullptr;
    float* rs = nullptr;  // per-row scales / batch-wide scalar
    float* ws = nullptr;
    float* bm = nullptr;  // batch-wide reduction scratch
    float* am = nullptr;
    half* y = nullptr;
    int grid = 0;
    Dev() {
        grid = ((kMaxM * kK / 4) + 255) / 256;
        EXPECT_EQ(cudaMalloc(&x, size_t(kMaxM) * kK * sizeof(half)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&x8, size_t(kMaxM) * kK), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&w, size_t(kN) * kK), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&rs, kMaxM * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&ws, sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&bm, grid * sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&am, sizeof(float)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&y, size_t(kMaxM) * kN * sizeof(half)), cudaSuccess);
        const std::vector<uint8_t> hw = random_e4m3(size_t(kN) * kK, 0x77u);
        EXPECT_EQ(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(ws, &kWScale, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    }
    ~Dev() {
        for (void* p :
             {static_cast<void*>(x), static_cast<void*>(x8), static_cast<void*>(w), static_cast<void*>(rs),
              static_cast<void*>(ws), static_cast<void*>(bm), static_cast<void*>(am), static_cast<void*>(y)})
            cudaFree(p);
    }
    void upload(const std::vector<half>& h, int M) {
        ASSERT_EQ(cudaMemcpy(x, h.data(), size_t(M) * kK * sizeof(half), cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
    std::vector<uint16_t> row0() const {
        std::vector<uint16_t> r(kN);
        EXPECT_EQ(cudaMemcpy(r.data(), y, kN * sizeof(half), cudaMemcpyDeviceToHost), cudaSuccess);
        return r;
    }
};

enum class Route { kSmallM, kTile, kTileStreamK };

// Row 0 of y = quant(x) W^T for M rows of `h`. per_row: per-token scales; else one batch-wide absmax scale.
std::vector<uint16_t> run(Dev& d, const std::vector<half>& h, int M, bool per_row, Route route) {
    d.upload(h, M);
    if (per_row) {
        EXPECT_TRUE(quantize_fp8_act_rows(d.x, d.x8, M, kK, d.rs, nullptr));
    } else {
        calibrate_and_quantize_fp8_async(d.x, d.x8, int64_t(M) * kK, d.bm, d.grid, d.am, d.rs, nullptr);
    }
    if (route == Route::kSmallM) {
        EXPECT_TRUE(gemm_fp8_smallm(d.x8, d.w, d.rs, d.ws, 0.0f, d.y, M, kN, kK, nullptr, per_row));
    } else {
        static void* ws = nullptr;
        const size_t ws_bytes = gemm_tile_streamk_workspace_bytes();
        if (route == Route::kTileStreamK && !ws)
            EXPECT_EQ(cudaMalloc(&ws, ws_bytes), cudaSuccess);
        const bool sk = route == Route::kTileStreamK;
        EXPECT_TRUE(gemm_fp8_tile(d.x8, d.w, d.rs, d.ws, 0.0f, d.y, M, kN, kK, nullptr, -1, sk ? ws : nullptr,
                                  sk ? ws_bytes : 0, per_row));
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return d.row0();
}

int differing(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    int n = 0;
    for (size_t i = 0; i < a.size(); ++i)
        n += a[i] != b[i];
    return n;
}

}  // namespace

// Codes and scales equal fp8_rows::row_scale / encode on the host, row by row.
TEST(Fp8ActRowsTest, QuantizeMatchesHostRule) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    Dev d;
    Batches bt;
    constexpr int M = 37;
    d.upload(bt.b, M);
    ASSERT_TRUE(quantize_fp8_act_rows(d.x, d.x8, M, kK, d.rs, nullptr));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<uint8_t> q(size_t(M) * kK);
    std::vector<float> sc(M);
    ASSERT_EQ(cudaMemcpy(q.data(), d.x8, q.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(sc.data(), d.rs, M * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    int bad = 0;
    for (int r = 0; r < M; ++r) {
        float m = 0.0f;
        for (int k = 0; k < kK; ++k)
            m = std::max(m, std::fabs(__half2float(bt.b[size_t(r) * kK + k])));
        // Device division may be the fast-math one: 1 ulp from the host's.
        const float s = fp8_rows::row_scale(m);
        EXPECT_NEAR(sc[r], s, 2e-7f * s) << "row " << r;
        // Codes from the device scale; a reciprocal 1 ulp off may move a tie by one code, never more.
        const float inv = 1.0f / sc[r];
        for (int k = 0; k < kK; ++k) {
            const uint8_t got = q[size_t(r) * kK + k];
            const uint8_t want = fp8_rows::encode(__half2float(bt.b[size_t(r) * kK + k]), inv);
            bad += (got & 0x80) != (want & 0x80) && (got & 0x7f) && (want & 0x7f);
            bad += std::abs(int(got & 0x7f) - int(want & 0x7f)) > 1;
        }
    }
    EXPECT_EQ(bad, 0);
    // Refusals: K % 8, misaligned input.
    EXPECT_FALSE(quantize_fp8_act_rows(d.x, d.x8, 2, 4092, d.rs, nullptr));
    EXPECT_FALSE(quantize_fp8_act_rows(d.x + 1, d.x8, 2, 64, d.rs, nullptr));
}

// The defect and its fix on every own route: same row 0, other neighbours. Batch-wide absmax moves row 0
// (control: must differ, else the input proves nothing); per-token scales keep it bit-equal.
TEST(Fp8ActRowsTest, RowBitsIndependentOfNeighbours) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    Dev d;
    Batches bt;
    struct Case {
        Route route;
        int M;
        const char* name;
    } const cases[] = {{Route::kSmallM, kFp8SmallMMaxM, "small-M M=32"},
                       {Route::kSmallM, 5, "small-M M=5"},
                       {Route::kTile, 64, "tile M=64"},
                       {Route::kTile, kMaxM, "tile M=600"},
                       {Route::kTileStreamK, kMaxM, "tile stream-K M=600"}};
    for (const Case& c : cases) {
        const std::vector<uint16_t> pa = run(d, bt.a, c.M, true, c.route);
        const std::vector<uint16_t> pb = run(d, bt.b, c.M, true, c.route);
        EXPECT_EQ(differing(pa, pb), 0) << c.name << ": per-token scales, row 0 moved with its neighbours";
        if (c.M > 7) {  // the outlier row is in the batch
            const std::vector<uint16_t> ga = run(d, bt.a, c.M, false, c.route);
            const std::vector<uint16_t> gb = run(d, bt.b, c.M, false, c.route);
            EXPECT_GT(differing(ga, gb), kN / 2) << c.name << ": control, batch-wide absmax must move row 0";
        }
    }
    // Data-parallel tile (runtime.deterministic): row 0 also equal across M.
    EXPECT_EQ(differing(run(d, bt.b, 64, true, Route::kTile), run(d, bt.b, kMaxM, true, Route::kTile)), 0);
    // The small-M kernel: row 0 equal across M (fixed K order).
    EXPECT_EQ(differing(run(d, bt.b, 9, true, Route::kSmallM), run(d, bt.b, 32, true, Route::kSmallM)), 0);
}

// Accuracy on the outlier batch: per-token row 0 error vs the FP16 reference at most the batch-wide one
// (whose scale the outlier row inflates 4096x), and within 2 % of the row's largest output.
TEST(Fp8ActRowsTest, PerRowNoWorseThanBatchWide) {
    if (!has_device())
        GTEST_SKIP() << "no CUDA device";
    Dev d;
    Batches bt;
    std::vector<uint8_t> hw(size_t(kN) * kK);
    ASSERT_EQ(cudaMemcpy(hw.data(), d.w, hw.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    std::vector<double> ref(kN);
    double ref_max = 0.0;
    for (int n = 0; n < kN; ++n) {
        double s = 0.0;
        for (int k = 0; k < kK; ++k)
            s += double(__half2float(bt.b[k])) * double(fp8_rows::decode(hw[size_t(n) * kK + k]));
        ref[n] = s * kWScale;
        ref_max = std::max(ref_max, std::fabs(ref[n]));
    }
    auto err = [&](const std::vector<uint16_t>& y) {
        double e = 0.0;
        for (int n = 0; n < kN; ++n) {
            half h;
            std::memcpy(&h, &y[n], sizeof(h));
            e += std::fabs(double(__half2float(h)) - ref[n]);
        }
        return e / kN;
    };
    for (Route r : {Route::kSmallM, Route::kTile}) {
        const int M = r == Route::kSmallM ? kFp8SmallMMaxM : kMaxM;
        const double e_row = err(run(d, bt.b, M, true, r));
        const double e_batch = err(run(d, bt.b, M, false, r));
        printf("[fp8-act-rows] route %d M=%d mean |err| per-token %.4e batch-wide %.4e (ref max %.3f)\n",
               static_cast<int>(r), M, e_row, e_batch, ref_max);
        EXPECT_LT(e_row, e_batch);
        EXPECT_LT(e_row, 0.02 * ref_max);
    }
}
