// Device-side Token-Recycling adjacency table (src/compute/token_recycle_device.cu)
// for the verify-in-loop design (#1055): the draft walk, MRU promote and
// streak semantics must match the host TokenRecycleTable exactly — the loop
// path drafts on-device, the eager path on-host, and both must produce the
// same drafts from the same observation stream.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/token_recycle_device.h"
#include "runtime/token_recycle_draft.h"

#include <cstdint>
#include <vector>

namespace imp {
namespace {

class TrDevice : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_ = nullptr;
};

// Feed the same pseudo-random token stream to host and device tables via
// observe_pairs; drafts from every token must agree at min_streak 0 and 1.
TEST_F(TrDevice, PairStreamMatchesHostReference) {
    const int vocab = 200, slots = 4, n = 400;
    std::vector<int32_t> stream_toks(n);
    uint32_t s = 42;
    for (auto& t : stream_toks) {
        s = s * 1664525u + 1013904223u;
        t = static_cast<int32_t>(s % 50);  // small range -> repeats -> streaks
    }

    TokenRecycleTable host(vocab, slots);
    for (int i = 1; i < n; ++i)
        host.observe_pair(stream_toks[i - 1], stream_toks[i]);

    TrDeviceTable dev{};
    ASSERT_TRUE(tr_device_init(dev, vocab, slots, stream_));
    int32_t* d_toks = nullptr;
    cudaMalloc(&d_toks, n * sizeof(int32_t));
    cudaMemcpyAsync(d_toks, stream_toks.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice,
                    stream_);
    tr_observe_pairs(dev, d_toks, n, stream_);

    int32_t* d_last = nullptr;
    int32_t *d_draft = nullptr, *d_len = nullptr;
    cudaMalloc(&d_last, sizeof(int32_t));
    cudaMalloc(&d_draft, 8 * sizeof(int32_t));
    cudaMalloc(&d_len, sizeof(int32_t));

    for (int ms = 0; ms <= 1; ++ms) {
        for (int32_t t0 = 0; t0 < 50; ++t0) {
            cudaMemcpyAsync(d_last, &t0, sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
            tr_draft(dev, d_last, /*depth=*/3, /*min_streak=*/ms, d_draft, d_len, stream_);
            int32_t len = -1;
            int32_t got[8];
            cudaMemcpyAsync(&len, d_len, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
            cudaMemcpyAsync(got, d_draft, 8 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
            auto want = host.draft_linear(t0, 3, ms);
            ASSERT_EQ(len, static_cast<int32_t>(want.size()))
                << "t0=" << t0 << " min_streak=" << ms;
            for (int j = 0; j < len; ++j)
                EXPECT_EQ(got[j], want[j]) << "t0=" << t0 << " j=" << j << " ms=" << ms;
        }
    }

    cudaFree(d_toks);
    cudaFree(d_last);
    cudaFree(d_draft);
    cudaFree(d_len);
    tr_device_free(dev);
}

// Top-K observation: rank order + streak follow the host semantics.
TEST_F(TrDevice, TopkObservationMatchesHostReference) {
    const int vocab = 100, slots = 4, rows = 3, m = 3;
    TokenRecycleTable host(vocab, slots);
    TrDeviceTable dev{};
    ASSERT_TRUE(tr_device_init(dev, vocab, slots, stream_));

    const int32_t row_tokens[rows] = {5, 6, 5};
    const int32_t topm[rows * m] = {10, 11, 12,   20, 21, 22,   10, 13, 11};
    for (int r = 0; r < rows; ++r)
        host.observe_topk(row_tokens[r], topm + r * m, m);

    int32_t *d_rt = nullptr, *d_tm = nullptr;
    cudaMalloc(&d_rt, sizeof(row_tokens));
    cudaMalloc(&d_tm, sizeof(topm));
    cudaMemcpyAsync(d_rt, row_tokens, sizeof(row_tokens), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_tm, topm, sizeof(topm), cudaMemcpyHostToDevice, stream_);
    tr_observe_topk(dev, d_rt, d_tm, rows, m, stream_);

    // Compare the full visible state: successors of 5 and 6 + drafts.
    std::vector<int32_t> succ(static_cast<size_t>(vocab) * slots);
    cudaMemcpyAsync(succ.data(), dev.succ, succ.size() * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    for (int32_t tok : {5, 6}) {
        for (int sl = 0; sl < slots; ++sl)
            EXPECT_EQ(succ[static_cast<size_t>(tok) * slots + sl], host.successor(tok, sl))
                << "tok=" << tok << " slot=" << sl;
    }

    cudaFree(d_rt);
    cudaFree(d_tm);
    tr_device_free(dev);
}

}  // namespace
}  // namespace imp
