// LFM2 short conv (shortconv_forward): chunked prefill and batched decode against a host reference
// of y_t = C_t * sum_k w[c,k] * (B*x)_{t-(L-1)+k}, window carried across calls.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "compute/shortconv.h"
#include "test_cuda_skip.h"

using namespace imp;

namespace {

constexpr int kH = 96, kL = 3;

float lcg(uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return static_cast<float>((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

// Host reference over a whole sequence of `rows` [T, 3H] (already fp16-rounded).
std::vector<float> reference(const std::vector<half>& rows, const std::vector<half>& w, int T) {
    std::vector<float> y(static_cast<size_t>(T) * kH);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < kH; ++c) {
            double acc = 0.0;
            for (int k = 0; k < kL; ++k) {
                const int j = t - (kL - 1) + k;
                if (j < 0)
                    continue;
                const half* r = rows.data() + static_cast<size_t>(j) * 3 * kH;
                acc += __half2float(w[c * kL + k]) * __half2float(r[c]) * __half2float(r[2 * kH + c]);
            }
            y[static_cast<size_t>(t) * kH + c] = static_cast<float>(acc) *
                                                 __half2float(rows[static_cast<size_t>(t) * 3 * kH + kH + c]);
        }
    return y;
}

}  // namespace

TEST(ShortConv, ChunkedPrefillThenBatchedDecodeMatchReference) {
    SKIP_IF_NO_CUDA();
    constexpr int kSeq = 3, kT = 11, kChunk = 7;  // 7 + 4 rows, then one decode row per sequence
    uint32_t seed = 12345;
    std::vector<half> w(kH * kL);
    for (auto& v : w)
        v = __float2half(lcg(seed));
    std::vector<std::vector<half>> rows(kSeq, std::vector<half>(static_cast<size_t>(kT + 1) * 3 * kH));
    for (auto& r : rows)
        for (auto& v : r)
            v = __float2half(lcg(seed));

    const int64_t stride = 4096;  // bytes between slots, > kH * kL * 4
    const int slots[kSeq] = {2, 0, 1};
    void* pool = nullptr;
    half *d_w = nullptr, *d_rows = nullptr, *d_y = nullptr;
    int* d_slots = nullptr;
    ASSERT_EQ(cudaMalloc(&pool, stride * kSeq), cudaSuccess);
    ASSERT_EQ(cudaMemset(pool, 0, stride * kSeq), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w, w.size() * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_rows, static_cast<size_t>(kT + 1) * 3 * kH * kSeq * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, static_cast<size_t>(kT + 1) * kH * kSeq * sizeof(half)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_slots, sizeof(slots)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_w, w.data(), w.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_slots, slots, sizeof(slots), cudaMemcpyHostToDevice), cudaSuccess);

    std::vector<std::vector<float>> got(kSeq, std::vector<float>(static_cast<size_t>(kT + 1) * kH));
    std::vector<half> buf(static_cast<size_t>(kT) * kH);
    // Prefill per sequence in two chunks, each on its own window.
    for (int s = 0; s < kSeq; ++s) {
        void* win = static_cast<char*>(pool) + slots[s] * stride;
        for (int t0 : {0, kChunk}) {
            const int n = (t0 == 0) ? kChunk : kT - kChunk;
            ASSERT_EQ(cudaMemcpy(d_rows, rows[s].data() + static_cast<size_t>(t0) * 3 * kH,
                                 static_cast<size_t>(n) * 3 * kH * sizeof(half), cudaMemcpyHostToDevice),
                      cudaSuccess);
            shortconv_forward(win, nullptr, 0, d_rows, d_w, d_y, 1, n, kH, kL, nullptr);
            ASSERT_EQ(cudaMemcpy(buf.data(), d_y, static_cast<size_t>(n) * kH * sizeof(half),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
            for (int i = 0; i < n * kH; ++i)
                got[s][static_cast<size_t>(t0) * kH + i] = __half2float(buf[i]);
        }
    }
    // One batched decode step: row kT of every sequence, windows selected through the slot table.
    std::vector<half> dec(static_cast<size_t>(kSeq) * 3 * kH);
    for (int s = 0; s < kSeq; ++s)
        std::copy_n(rows[s].data() + static_cast<size_t>(kT) * 3 * kH, 3 * kH,
                    dec.data() + static_cast<size_t>(s) * 3 * kH);
    ASSERT_EQ(cudaMemcpy(d_rows, dec.data(), dec.size() * sizeof(half), cudaMemcpyHostToDevice), cudaSuccess);
    shortconv_forward(pool, d_slots, stride, d_rows, d_w, d_y, kSeq, 1, kH, kL, nullptr);
    ASSERT_EQ(cudaMemcpy(buf.data(), d_y, static_cast<size_t>(kSeq) * kH * sizeof(half),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    for (int s = 0; s < kSeq; ++s)
        for (int c = 0; c < kH; ++c)
            got[s][static_cast<size_t>(kT) * kH + c] = __half2float(buf[static_cast<size_t>(s) * kH + c]);

    for (int s = 0; s < kSeq; ++s) {
        const auto want = reference(rows[s], w, kT + 1);
        for (size_t i = 0; i < want.size(); ++i)
            ASSERT_NEAR(got[s][i], want[i], 2e-3 + 2e-3 * std::fabs(want[i])) << "seq " << s << " elem " << i;
    }
    cudaFree(pool);
    cudaFree(d_w);
    cudaFree(d_rows);
    cudaFree(d_y);
    cudaFree(d_slots);
}
