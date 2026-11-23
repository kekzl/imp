// Inkling2 DSpark draft (exec/inkling_draft.h) on the real checkpoint: IMP_TEST_MODEL_INKLING, and
// IMP_TEST_INKLING_ORACLE for the reference continuation.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "exec/inkling_draft.h"
#include "exec/inkling_forward.h"
#include "inkling_fixture.h"
#include "quant/fp8_quant.h"

using namespace imp;
using namespace imp::inkling_test;

namespace {

// The checkpoint's draft on the reference continuation: fed the tokens the transformers reference
// generated, the lookahead confirms draft rows only where the draft named exactly those tokens.
TEST(InklingDraftModel, DraftsFollowTheReferenceContinuation) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const int V = fx.w.cfg.vocab, H = fx.w.cfg.hidden;
    // The LM head as the engine holds it: per-row E4M3 with row scales.
    void* d_w8 = nullptr;
    float* d_sc = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w8, static_cast<size_t>(V) * H), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sc, static_cast<size_t>(V) * sizeof(float)), cudaSuccess);
    quantize_fp8_rows_async(fx.w.unembed, d_w8, V, H, d_sc, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    long steps = 0, drafted = 0, accepted = 0, tokens = 0;
    {
        Stepper st(fx.w);
        InklingDraftWeights dw;
        ASSERT_TRUE(dw.load(model, fx.va, fx.w.embed_norm, InklingHeadView{d_w8, d_sc, V}, st.s));
        for (const char* name : {"paris", "sky"}) {
            std::vector<int32_t> t = read_ids(std::string(oracle) + "/" + name + "_prompt.json");
            const std::vector<int32_t> g = read_ids(std::string(oracle) + "/" + name + "_greedy.json");
            ASSERT_FALSE(t.empty());
            const int P = static_cast<int>(t.size());
            t.insert(t.end(), g.begin(), g.end());
            InklingDraft draft;
            InklingForward f;
            ASSERT_TRUE(f.init(fx.w, fx.va, 512, 4096));
            ASSERT_TRUE(draft.init(dw, fx.va, 4096, InklingForward::kGrid));
            ASSERT_TRUE(f.enable_lookahead(draft.hooks(), dw.cfg.taps, fx.w.embed));
            ASSERT_FALSE(run_rows(f, fx.w, std::vector<int32_t>(t.begin(), t.begin() + P), 0, {}).empty());
            for (int pos = P; pos < static_cast<int>(t.size()); ++pos)
                ASSERT_FALSE(st.step(f, t[static_cast<size_t>(pos)], pos).empty());
            const InklingForward::SpecStats s = f.spec_stats();
            std::printf("draft %s: %ld steps for %zu tokens, %ld draft rows run, %ld confirmed\n", name,
                        s.steps, g.size(), s.drafted, s.accepted);
            steps += s.steps, drafted += s.drafted, accepted += s.accepted,
                tokens += static_cast<long>(g.size());
            EXPECT_EQ(s.steps + s.accepted, static_cast<long>(g.size()));
        }
    }
    std::printf("draft: %.2f tokens per step, %.0f %% of the run draft rows confirmed\n",
                static_cast<double>(tokens) / static_cast<double>(steps),
                100.0 * static_cast<double>(accepted) / static_cast<double>(std::max(drafted, 1L)));
    // Measured: 57 of the 96 tokens come out of confirmed draft rows (33 + 24, 88 rows run); 42 without
    // the Markov head's correction.
    EXPECT_GE(accepted, 52);
    cudaFree(d_w8);
    cudaFree(d_sc);
}

// A drafted call's rows get their context K/V at once (on_rows after propose). Rows a later call brings
// for the same positions must replace them: the draft after 8 replaced rows names the tokens of a draft
// that never saw the first 8.
TEST(InklingDraftModel, ReplacedRowsLeaveNoTraceInTheDraft) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    if (!model)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const int V = fx.w.cfg.vocab, H = fx.w.cfg.hidden, cap = InklingForward::kGrid, P = 40, R = 8;
    void* d_w8 = nullptr;
    float *d_sc = nullptr, *d_feat = nullptr;
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w8, static_cast<size_t>(V) * H), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sc, static_cast<size_t>(V) * sizeof(float)), cudaSuccess);
    quantize_fp8_rows_async(fx.w.unembed, d_w8, V, H, d_sc, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    {
        InklingDraftWeights dw;
        ASSERT_TRUE(dw.load(model, fx.va, fx.w.embed_norm, InklingHeadView{d_w8, d_sc, V}, s));
        const size_t taps = dw.cfg.taps.size(), plane = static_cast<size_t>(cap) * H;
        ASSERT_EQ(cudaMalloc(&d_feat, taps * plane * sizeof(float)), cudaSuccess);
        std::vector<float> host(taps * plane);
        // n feature rows from seed at positions pos0...
        const auto feed = [&](InklingDraft& d, uint32_t seed, int n, int pos0) {
            for (size_t t = 0; t < taps; ++t)
                for (size_t i = 0; i < static_cast<size_t>(n) * H; ++i) {
                    seed = seed * 1664525u + 1013904223u;
                    host[t * plane + i] = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f;
                }
            cudaMemcpy(d_feat, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
            d.on_rows(d_feat, cap, n, pos0, s);
            cudaStreamSynchronize(s);
        };
        const int32_t anchor = 1000;
        const half* emb = fx.w.embed + static_cast<size_t>(anchor) * H;
        int32_t got[InklingForward::kGraphRows] = {}, want[InklingForward::kGraphRows] = {};
        int32_t skip[InklingForward::kGraphRows] = {};
        {
            InklingDraft d;
            ASSERT_TRUE(d.init(dw, fx.va, 4096, cap));
            feed(d, 1u, P, 0);
            (void)d.propose(anchor, emb, P, skip, s);
            feed(d, 2u, R, P);  // the drafted call's rows
            feed(d, 3u, R, P);  // the rows that replace them
            (void)d.propose(anchor, emb, P + R, got, s);
        }
        {
            InklingDraft d;
            ASSERT_TRUE(d.init(dw, fx.va, 4096, cap));
            feed(d, 1u, P, 0);
            (void)d.propose(anchor, emb, P, skip, s);
            feed(d, 3u, R, P);
            (void)d.propose(anchor, emb, P + R, want, s);
        }
        int same = 0, in_vocab = 0;
        for (int i = 0; i < InklingForward::kGraphRows - 1; ++i) {
            same += got[i] == want[i];
            in_vocab += want[i] >= 0 && want[i] < V;
        }
        std::printf("draft after 8 replaced rows: %d of 7 tokens equal a draft that never saw them\n", same);
        EXPECT_EQ(in_vocab, InklingForward::kGraphRows - 1);
        EXPECT_EQ(same, InklingForward::kGraphRows - 1);
    }
    cudaFree(d_feat);
    cudaFree(d_w8);
    cudaFree(d_sc);
    cudaStreamDestroy(s);
}

// A local draft layer keeps its context K/V in a ring instead of one row per position. Drafts along
// 3 ring lengths of context, fed in calls of cap rows and of 8 and once from back positions before the
// end, must name the tokens of a draft that holds every position (back < 0).
TEST(InklingDraftModel, LocalLayerRingMatchesFullContext) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    if (!model)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const int V = fx.w.cfg.vocab, H = fx.w.cfg.hidden, cap = InklingForward::kGrid, ctx = 16384;
    const int back = 3 * cap;
    void* d_w8 = nullptr;
    float *d_sc = nullptr, *d_feat = nullptr;
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_w8, static_cast<size_t>(V) * H), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sc, static_cast<size_t>(V) * sizeof(float)), cudaSuccess);
    quantize_fp8_rows_async(fx.w.unembed, d_w8, V, H, d_sc, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    {
        InklingDraftWeights dw;
        ASSERT_TRUE(dw.load(model, fx.va, fx.w.embed_norm, InklingHeadView{d_w8, d_sc, V}, s));
        const size_t taps = dw.cfg.taps.size(), plane = static_cast<size_t>(cap) * H;
        ASSERT_EQ(cudaMalloc(&d_feat, taps * plane * sizeof(float)), cudaSuccess);
        std::vector<float> host(taps * plane);
        const int ring = dw.cfg.window + dw.cfg.block + cap + back;
        ASSERT_LT(4 * ring, ctx);
        const size_t before = fx.va.allocated();
        InklingDraft ringed, full;
        ASSERT_TRUE(ringed.init(dw, fx.va, ctx, cap, back));
        const size_t ring_bytes = fx.va.allocated() - before;
        ASSERT_TRUE(full.init(dw, fx.va, ctx, cap, -1));
        const size_t full_bytes = fx.va.allocated() - before - ring_bytes;
        std::printf("draft context of %d positions: %.1f MiB with the ring, %.1f MiB without\n", ctx,
                    static_cast<double>(ring_bytes) / 1048576.0, static_cast<double>(full_bytes) / 1048576.0);
        EXPECT_LT(ring_bytes, full_bytes);
        const int32_t anchor = 1000;
        const half* emb = fx.w.embed + static_cast<size_t>(anchor) * H;
        uint32_t seed = 7u;
        int pos = 0, drafts = 0, same = 0, rows = 0;
        // Calls of cap rows, every third position range in calls of 8 with a draft before each. The last
        // range starts again back positions before the end, as a sequence that resumes there does.
        bool again = false;
        while (pos + cap + InklingForward::kGraphRows < 3 * ring + cap || !again) {
            if (pos + cap + InklingForward::kGraphRows >= 3 * ring + cap)
                again = true, pos -= back;
            const bool small = (pos / cap) % 3 == 2 && !again;
            for (int at = 0; at < cap;) {
                const int n = small ? 8 : cap;
                for (size_t t = 0; t < taps; ++t)
                    for (size_t i = 0; i < static_cast<size_t>(n) * H; ++i) {
                        seed = seed * 1664525u + 1013904223u;
                        host[t * plane + i] = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f;
                    }
                ASSERT_EQ(cudaMemcpy(d_feat, host.data(), host.size() * sizeof(float),
                                     cudaMemcpyHostToDevice),
                          cudaSuccess);
                ringed.on_rows(d_feat, cap, n, pos + at, s);
                full.on_rows(d_feat, cap, n, pos + at, s);
                at += n;
                int32_t got[InklingForward::kGraphRows] = {}, want[InklingForward::kGraphRows] = {};
                (void)ringed.propose(anchor, emb, pos + at, got, s);
                (void)full.propose(anchor, emb, pos + at, want, s);
                ++drafts;
                for (int i = 0; i < InklingForward::kGraphRows - 1; ++i) {
                    same += got[i] == want[i];
                    rows += want[i] >= 0 && want[i] < V;
                }
            }
            pos += cap;
        }
        std::printf("draft ring: %d of %d tokens of %d drafts up to position %d equal the full context's\n",
                    same, 7 * drafts, drafts, pos);
        EXPECT_GT(pos, 3 * ring - cap - back);
        EXPECT_EQ(rows, 7 * drafts);
        EXPECT_EQ(same, 7 * drafts);
    }
    cudaFree(d_feat);
    cudaFree(d_w8);
    cudaFree(d_sc);
    cudaStreamDestroy(s);
}

}  // namespace
