// tr_verify_step: the device accept+draft+stage kernel that forms the tail
// of the verify-in-loop conditional body (#1055, docs/plans/2026-07-23-
// verify-in-loop.md). Driven here WITHOUT a graph: the test stages chunk 0,
// then plays model "argmax" outputs into the buffer and steps the kernel,
// checking ring emission, position advance, next-chunk staging, adjacency
// updates and the exit reasons.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "compute/token_recycle_device.h"

#include <cstdint>
#include <vector>

namespace imp {
namespace {

constexpr int kPad = 4;   // bucket-4 chunk
constexpr int kM = 3;     // top-M harvest width

class TrVerifyStep : public ::testing::Test {
protected:
    void SetUp() override {
        cudaStreamCreate(&stream_);
        ASSERT_TRUE(tr_device_init(tab_, /*vocab=*/100, /*slots=*/4, stream_));
        cudaMalloc(&d_stage_, (3 * kPad + 3) * sizeof(int32_t));
        cudaMalloc(&d_argmax_, kPad * sizeof(int32_t));
        cudaMalloc(&d_topm_, kPad * kM * sizeof(int32_t));
        cudaMalloc(&d_emit_count_, sizeof(int32_t));
        cudaMalloc(&d_exit_reason_, sizeof(int32_t));
        cudaHostAlloc(&h_ring_, 64 * sizeof(int32_t), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_ring_, h_ring_, 0);
        cudaHostAlloc(&h_count_, sizeof(int32_t), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_count_mapped_, h_count_, 0);
        cudaMemsetAsync(d_emit_count_, 0, sizeof(int32_t), stream_);
        cudaMemsetAsync(d_exit_reason_, 0, sizeof(int32_t), stream_);
        *h_count_ = 0;
    }
    void TearDown() override {
        tr_device_free(tab_);
        cudaFree(d_stage_);
        cudaFree(d_argmax_);
        cudaFree(d_topm_);
        cudaFree(d_emit_count_);
        cudaFree(d_exit_reason_);
        cudaFreeHost(h_ring_);
        cudaFreeHost(h_count_);
        cudaStreamDestroy(stream_);
    }

    TrLoopView view() {
        TrLoopView v{};
        v.tab = tab_;
        v.tokens = d_stage_;
        v.positions = d_stage_ + kPad;
        v.row_ctx_lens = d_stage_ + 2 * kPad;
        v.ctx_len = d_stage_ + 3 * kPad;
        v.past_len = d_stage_ + 3 * kPad + 1;
        v.chunk_len = d_stage_ + 3 * kPad + 2;
        v.argmax = d_argmax_;
        v.topm = d_topm_;
        v.ring = d_ring_;
        v.ring_count_mapped = d_count_mapped_;
        v.emit_count = d_emit_count_;
        v.exit_reason = d_exit_reason_;
        return v;
    }

    TrLoopParams params() {
        TrLoopParams p{};
        p.chunk_pad = kPad;
        p.depth = 3;
        p.min_streak = 0;
        p.topm = kM;
        p.eos_id = 99;
        p.token_limit = 64;
        p.ctx_ceiling = 4096;
        return p;
    }

    // Host-side stage of the first chunk (the launch-time seed).
    void seed_chunk(int32_t t0, const std::vector<int32_t>& draft, int p0) {
        int32_t stage[3 * kPad + 3];
        const int L = 1 + static_cast<int>(draft.size());
        for (int i = 0; i < kPad; ++i) {
            stage[i] = (i == 0) ? t0 : (i <= static_cast<int>(draft.size()) ? draft[i - 1] : t0);
            stage[kPad + i] = p0 + i;
            stage[2 * kPad + i] = (i < L) ? (p0 + i + 1) : 1;
        }
        stage[3 * kPad] = p0 + kPad;
        stage[3 * kPad + 1] = p0;
        stage[3 * kPad + 2] = L;
        cudaMemcpyAsync(d_stage_, stage, sizeof(stage), cudaMemcpyHostToDevice, stream_);
    }

    void set_argmax(const std::vector<int32_t>& v) {
        int32_t a[kPad] = {0, 0, 0, 0};
        for (size_t i = 0; i < v.size() && i < kPad; ++i) a[i] = v[i];
        cudaMemcpyAsync(d_argmax_, a, sizeof(a), cudaMemcpyHostToDevice, stream_);
        int32_t tm[kPad * kM];
        for (int i = 0; i < kPad * kM; ++i) tm[i] = 1 + (i % 7);  // arbitrary valid ids
        cudaMemcpyAsync(d_topm_, tm, sizeof(tm), cudaMemcpyHostToDevice, stream_);
    }

    int32_t stage_at(int idx) {
        int32_t v;
        cudaMemcpy(&v, d_stage_ + idx, sizeof(int32_t), cudaMemcpyDeviceToHost);
        return v;
    }
    int32_t exit_reason() {
        int32_t v;
        cudaMemcpy(&v, d_exit_reason_, sizeof(int32_t), cudaMemcpyDeviceToHost);
        return v;
    }

    cudaStream_t stream_ = nullptr;
    TrDeviceTable tab_{};
    int32_t* d_stage_ = nullptr;
    int32_t* d_argmax_ = nullptr;
    int32_t* d_topm_ = nullptr;
    int32_t* d_emit_count_ = nullptr;
    int32_t* d_exit_reason_ = nullptr;
    int32_t* h_ring_ = nullptr;
    int32_t* d_ring_ = nullptr;
    int32_t* h_count_ = nullptr;
    int32_t* d_count_mapped_ = nullptr;
};

// Full accept: draft [2,3] all match, bonus 4 -> emit [2,3,4]; next chunk
// staged from token 4 with the chain the adjacency has learned.
TEST_F(TrVerifyStep, FullAcceptEmitsAndStagesNext) {
    // Teach the table 4 -> 5 -> 6 so the next draft exists.
    const int32_t chain[3] = {4, 5, 6};
    int32_t* d_chain;
    cudaMalloc(&d_chain, sizeof(chain));
    cudaMemcpyAsync(d_chain, chain, sizeof(chain), cudaMemcpyHostToDevice, stream_);
    tr_observe_pairs(tab_, d_chain, 3, stream_);

    seed_chunk(/*t0=*/1, {2, 3}, /*p0=*/10);  // chunk rows: [1,2,3,pad] L=3
    set_argmax({2, 3, 4, 0});                 // rows agree, bonus = 4
    tr_verify_step(view(), params(), /*no_handle=*/true, stream_);
    cudaStreamSynchronize(stream_);

    EXPECT_EQ(*h_count_, 3);
    EXPECT_EQ(h_ring_[0], 2);
    EXPECT_EQ(h_ring_[1], 3);
    EXPECT_EQ(h_ring_[2], 4);
    EXPECT_EQ(exit_reason(), 0);              // loop continues
    EXPECT_EQ(stage_at(0), 4);                // next t0 = bonus
    EXPECT_EQ(stage_at(1), 5);                // next draft from adjacency
    EXPECT_EQ(stage_at(kPad), 13);            // next p0 = 10 + 3
    EXPECT_EQ(stage_at(3 * kPad + 1), 13);    // past_len
    cudaFree(d_chain);
}

// Partial accept: draft [2,3], model diverges at row 1 -> emit [2, 7].
TEST_F(TrVerifyStep, PartialAcceptStopsAtDivergence) {
    const int32_t chain[2] = {7, 8};
    int32_t* d_chain;
    cudaMalloc(&d_chain, sizeof(chain));
    cudaMemcpyAsync(d_chain, chain, sizeof(chain), cudaMemcpyHostToDevice, stream_);
    tr_observe_pairs(tab_, d_chain, 2, stream_);

    seed_chunk(1, {2, 3}, 10);
    set_argmax({2, 7, 0, 0});  // row0 matches draft(2), row1 says 7 != 3
    tr_verify_step(view(), params(), true, stream_);
    cudaStreamSynchronize(stream_);

    EXPECT_EQ(*h_count_, 2);
    EXPECT_EQ(h_ring_[0], 2);
    EXPECT_EQ(h_ring_[1], 7);
    EXPECT_EQ(exit_reason(), 0);
    EXPECT_EQ(stage_at(0), 7);              // next t0 = the diverging token
    EXPECT_EQ(stage_at(1), 8);              // drafted from 7 -> 8
    EXPECT_EQ(stage_at(3 * kPad + 1), 12);  // p0 10 + 2 emitted
    cudaFree(d_chain);
}

// Miss exit: bonus token has no successor -> emit, then exit_reason=miss.
TEST_F(TrVerifyStep, MissExitsLoop) {
    seed_chunk(1, {2}, 10);
    set_argmax({2, 50, 0, 0});  // accept 2, bonus 50 (no successors known)
    tr_verify_step(view(), params(), true, stream_);
    cudaStreamSynchronize(stream_);
    EXPECT_EQ(*h_count_, 2);
    EXPECT_EQ(exit_reason(), 1);  // 1 = draft miss
}

// EOS mid-chunk: truncates the emission at EOS and exits.
TEST_F(TrVerifyStep, EosTruncatesAndExits) {
    seed_chunk(1, {99, 3}, 10);   // draft contains EOS(99)
    set_argmax({99, 0, 0, 0});    // model emits EOS at row 0
    tr_verify_step(view(), params(), true, stream_);
    cudaStreamSynchronize(stream_);
    EXPECT_EQ(*h_count_, 1);
    EXPECT_EQ(h_ring_[0], 99);
    EXPECT_EQ(exit_reason(), 2);  // 2 = stop token
}

// Token budget exit.
TEST_F(TrVerifyStep, TokenLimitExits) {
    seed_chunk(1, {2, 3}, 10);
    set_argmax({2, 3, 4, 0});
    auto p = params();
    p.token_limit = 2;  // only 2 tokens allowed
    tr_verify_step(view(), p, true, stream_);
    cudaStreamSynchronize(stream_);
    EXPECT_EQ(*h_count_, 2);      // truncated at the budget
    EXPECT_EQ(exit_reason(), 3);  // 3 = budget
}

}  // namespace
}  // namespace imp
