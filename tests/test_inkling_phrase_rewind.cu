// Inkling2-Loop banned-phrase rewind (B-111 S6): the engine snapshots a slot's conv state
// (InklingForward::conv_snapshot) or takes the slot's own grid checkpoint, seeks back to it
// (InklingForward::seek) and re-forwards the kept tokens before it resamples the banned position. Needs
// IMP_TEST_MODEL_INKLING and IMP_TEST_INKLING_ORACLE (tests/test_inkling_parity.cu); runs in the
// test-compute process that loads the checkpoint once.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "exec/inkling_forward.h"
#include "inkling_fixture.h"

using namespace imp;
using namespace imp::inkling_test;

namespace {

bool have_model() { return std::getenv("IMP_TEST_MODEL_INKLING") && std::getenv("IMP_TEST_INKLING_ORACLE"); }

// Lookahead drafts from a plan: anchor position -> tokens proposed after it (tests/test_inkling_parity.cu).
InklingSpecHooks plan_hooks(const std::map<int, std::vector<int32_t>>& plan) {
    InklingSpecHooks h;
    h.on_rows = [](const float*, int, int, int, cudaStream_t) {};
    h.propose = [&plan](int32_t, const half*, int pos, int32_t* out, cudaStream_t) {
        const auto it = plan.find(pos);
        if (it == plan.end())
            return 0;
        std::copy(it->second.begin(), it->second.end(), out);
        return static_cast<int>(it->second.size());
    };
    return h;
}

int32_t argmax(const std::vector<float>& v, int32_t skip = -1) {
    int32_t am = skip == 0 ? 1 : 0;
    for (int32_t i = 0; i < static_cast<int32_t>(v.size()); ++i)
        if (i != skip && v[static_cast<size_t>(i)] > v[static_cast<size_t>(am)])
            am = i;
    return am;
}

}  // namespace

// The serve path's decode steps with the lookahead on. State 1 is taken while a stored lookahead call serves
// its rows (anchor 557, 3 of 7 drafts confirmed, so its conv state also holds 4 rejected rows); state 2 is
// the slot's own checkpoint at 768 in plain one-row steps. Seeking back to either from 820 and stepping the
// same tokens again gives the rows of the first run, bit for bit.
TEST(InklingPhraseRewind, SeekReplaysTheRowsOfTheFirstRun) {
    if (!have_model())
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(std::getenv("IMP_TEST_MODEL_INKLING")));
    const std::vector<int32_t> p = read_ids(std::string(std::getenv("IMP_TEST_INKLING_ORACLE")) +
                                            "/paris_prompt.json");
    ASSERT_FALSE(p.empty());
    const int P = 525, N = 820, kServed = 561, kGridPos = 768;
    std::vector<int32_t> t;
    while (static_cast<int>(t.size()) < N + 8)
        t.insert(t.end(), p.begin(), p.end());
    std::map<int, std::vector<int32_t>> plan;
    for (const int a : {541, 557}) {
        std::vector<int32_t> d(t.begin() + a + 1, t.begin() + a + 8);
        d[3] = t[0] == d[3] ? t[1] : t[0];  // rows 1..3 confirmed, the 4th rejected
        plan[a] = d;
    }
    const std::vector<int> taps = {5, 23, 41, 48};
    InklingForward f;
    ASSERT_TRUE(f.init(fx.w, fx.va, 512, 4096));
    ASSERT_TRUE(f.enable_lookahead(plan_hooks(plan), taps, fx.w.embed));
    ASSERT_FALSE(run_rows(f, fx.w, std::vector<int32_t>(t.begin(), t.begin() + P), 0, {256}).empty());
    float* snap = nullptr;
    ASSERT_EQ(cudaMalloc(&snap, f.conv_state_bytes()), cudaSuccess);
    std::vector<std::vector<float>> first(static_cast<size_t>(N));
    {
        Stepper st(fx.w);
        for (int pos = P; pos < N; ++pos) {
            if (pos == kServed) {
                ASSERT_EQ(f.position(), kServed);
                ASSERT_EQ(f.look_row(), 3) << "the step before was the 3rd row served from the stored call";
                f.conv_snapshot(snap, st.s);
            }
            first[static_cast<size_t>(pos)] = st.step(f, t[static_cast<size_t>(pos)], pos);
            ASSERT_FALSE(first[static_cast<size_t>(pos)].empty());
        }
        EXPECT_EQ(f.spec_stats().accepted, 6);
    }
    EXPECT_EQ(f.checkpoint_state(kGridPos + 1), nullptr);  // off the grid
    EXPECT_FALSE(f.seek(N + 1, snap, nullptr));            // ahead of the sequence
    EXPECT_EQ(f.position(), N);
    for (const int back : {kGridPos, kServed}) {
        Stepper st(fx.w);
        const float* from = back == kGridPos ? f.checkpoint_state(kGridPos) : snap;
        ASSERT_NE(from, nullptr);
        ASSERT_TRUE(f.seek(back, from, st.s));
        ASSERT_EQ(f.position(), back);
        int diff = 0;
        for (int pos = back; pos < N; ++pos) {
            const std::vector<float> r = st.step(f, t[static_cast<size_t>(pos)], pos);
            ASSERT_FALSE(r.empty());
            diff += std::memcmp(r.data(), first[static_cast<size_t>(pos)].data(), r.size() * sizeof(float)) !=
                    0;
        }
        std::printf("seek to %d: %d of %d replayed rows differ from the first run\n", back, diff, N - back);
        EXPECT_EQ(diff, 0) << "seek to " << back;
    }
    cudaFree(snap);
}

// What the engine does with a ban, on the forward's own logits: greedy tokens from the paris prompt, the
// token at output position 12 banned. Seek to the state after 8 output tokens, re-forward tokens 8..11:
// their logits are the first run's bit for bit (the kept text), and position 12 resamples to the
// first run's runner-up there, never the banned token.
TEST(InklingPhraseRewind, BannedTokenResamplesAfterTheKeptText) {
    if (!have_model())
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(std::getenv("IMP_TEST_MODEL_INKLING")));
    const std::vector<int32_t> ids = read_ids(std::string(std::getenv("IMP_TEST_INKLING_ORACLE")) +
                                              "/paris_prompt.json");
    ASSERT_FALSE(ids.empty());
    const int T = static_cast<int>(ids.size()), V = fx.w.cfg.vocab, kSnap = 8, kBan = 12, kOut = 16;
    InklingForward fwd;
    ASSERT_TRUE(fwd.init(fx.w, fx.va, std::max(T, 512), 4096));
    int32_t* d_ids = nullptr;
    float *d_logits = nullptr, *snap = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ids, ids.size() * sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_logits, static_cast<size_t>(V) * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&snap, fwd.conv_state_bytes()), cudaSuccess);
    cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    ASSERT_TRUE(fwd.forward(d_ids, T, d_logits, nullptr));
    // logits[i]: the distribution output token i is drawn from.
    std::vector<std::vector<float>> logits;
    std::vector<int32_t> out;
    const auto next = [&](int32_t tok) {
        cudaMemcpy(d_ids, &tok, sizeof tok, cudaMemcpyHostToDevice);
        return fwd.forward(d_ids, 1, d_logits, nullptr);
    };
    const auto read = [&] {
        std::vector<float> l(static_cast<size_t>(V));
        cudaMemcpy(l.data(), d_logits, l.size() * sizeof(float), cudaMemcpyDeviceToHost);
        return l;
    };
    for (int i = 0; i < kOut; ++i) {
        logits.push_back(read());
        out.push_back(argmax(logits.back()));
        if (fwd.position() == T + kSnap)
            fwd.conv_snapshot(snap, nullptr);
        ASSERT_TRUE(next(out.back()));
    }
    const int32_t banned = out[kBan];
    // The rewind: back to the state after kSnap output tokens, the kept tokens kSnap..kBan-1 again.
    ASSERT_TRUE(fwd.seek(T + kSnap, snap, nullptr));
    int diff = 0;
    std::vector<float> at_ban;
    for (int i = kSnap; i < kBan; ++i) {
        ASSERT_TRUE(next(out[static_cast<size_t>(i)]));
        at_ban = read();
        const std::vector<float>& want = logits[static_cast<size_t>(i) + 1];
        diff += std::memcmp(at_ban.data(), want.data(), want.size() * sizeof(float)) != 0;
    }
    EXPECT_EQ(diff, 0) << "replayed logits differ from the first run";
    const int32_t again = argmax(at_ban, banned);
    std::printf("ban: token %d at output %d resampled to %d (first run's runner-up %d)\n", banned, kBan,
                again, argmax(logits[kBan], banned));
    EXPECT_NE(again, banned);
    EXPECT_EQ(again, argmax(logits[kBan], banned));
    cudaFree(d_ids);
    cudaFree(d_logits);
    cudaFree(snap);
}
