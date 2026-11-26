// Inkling2-Loop prefix reuse across sequence slots (B-111 S4): one slot's state at a checkpoint through a
// RecurrentSnapshotStore entry into another slot. Needs IMP_TEST_MODEL_INKLING and IMP_TEST_INKLING_ORACLE
// (tests/test_inkling_parity.cu); runs in the test-compute process that loads the checkpoint once.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "exec/inkling_draft.h"
#include "exec/inkling_forward.h"
#include "exec/inkling_kv_layout.h"
#include "exec/inkling_slab_layout.h"
#include "inkling_fixture.h"
#include "memory/recurrent_snapshot_store.h"
#include "memory/vram_allocator.h"
#include "quant/fp8_quant.h"

using namespace imp;
using namespace imp::inkling_test;

namespace {

// Save on one slot, resume on another (B-111 S4): slot A runs a (calls [0, 256), [256, 300)), its state at
// 256 goes through a RecurrentSnapshotStore entry into slot B, which last ran another prompt on other pool
// blocks; B continues b (a's first 280 tokens, 30 others) from 256 over A's pool blocks. Same calls as a
// fresh run of b: bit-equal rows, from the device tier and again from the host tier after a demotion.
TEST(InklingCrossSlot, RestoredPrefixMatchesAFreshRun) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const std::vector<int32_t> sky = read_ids(std::string(oracle) + "/sky_prompt.json");
    const std::vector<int32_t> paris = read_ids(std::string(oracle) + "/paris_prompt.json");
    ASSERT_FALSE(sky.empty());
    ASSERT_FALSE(paris.empty());
    std::vector<int32_t> a, b, other;
    for (size_t i = 0; i < 300; ++i)
        a.push_back(sky[i % sky.size()]);
    b.assign(a.begin(), a.begin() + 280);
    for (size_t i = 0; i < 30; ++i)
        b.push_back(paris[i % paris.size()]);
    other.assign(a.rbegin(), a.rend());
    const int ctx = 4096;
    const InklingKvLayout l = inkling_kv_apps(fx.w.cfg, false);
    const size_t KD = static_cast<size_t>(fx.w.cfg.n_kv_heads) * fx.w.cfg.head_dim;
    std::vector<float> got[2], want;
    std::vector<void*> owned;
    {
        InklingForward A, B, fresh;
        ASSERT_TRUE(A.init(fx.w, fx.va, 512, ctx, nullptr, /*own_kv=*/false));
        ASSERT_TRUE(B.init(fx.w, fx.va, 512, ctx, nullptr, /*own_kv=*/false));
        ASSERT_TRUE(fresh.init(fx.w, fx.va, 512, ctx));
        // One pool of 2 x nb blocks: a's sequence in [0, nb), B's other prompt in [nb, 2 nb).
        const int nb = A.table_cap();
        std::vector<half*> k, v;
        for (int i = 0; i < 2 * l.n_global; ++i) {
            void* d = nullptr;
            const size_t bytes = 2 * static_cast<size_t>(nb) * 16 * KD * sizeof(half);
            ASSERT_EQ(cudaMalloc(&d, bytes), cudaSuccess);
            ASSERT_EQ(cudaMemset(d, 0, bytes), cudaSuccess);
            owned.push_back(d);
            (i % 2 ? v : k).push_back(static_cast<half*>(d));
        }
        std::vector<int> ids(2 * static_cast<size_t>(nb));
        std::iota(ids.begin(), ids.end(), 0);
        int* d_tbl = nullptr;
        ASSERT_EQ(cudaMalloc(&d_tbl, ids.size() * sizeof(int)), cudaSuccess);
        owned.push_back(d_tbl);
        ASSERT_EQ(cudaMemcpy(d_tbl, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_TRUE(A.bind_kv(k, v));
        ASSERT_TRUE(B.bind_kv(k, v));
        A.set_table(d_tbl, nb, nullptr);
        const auto foreign = [&] {  // B's rings and conv state hold another sequence
            B.set_table(d_tbl + nb, nb, nullptr);
            return !run_rows(B, fx.w, other, 0, {256}).empty();
        };
        ASSERT_TRUE(foreign());
        ASSERT_FALSE(run_rows(A, fx.w, a, 0, {256}).empty());
        EXPECT_EQ(B.match(b), 0);
        ASSERT_EQ(A.snapshot_pos(a), 256);
        std::vector<SnapshotPiece> pieces;
        ASSERT_TRUE(A.snapshot_pieces(256, pieces, nullptr));
        RecurrentSnapshotStore store;
        store.init(A.snapshot_bytes(), A.snapshot_bytes(), 2 * A.snapshot_bytes());
        ASSERT_TRUE(store.enabled());
        ASSERT_TRUE(store.save_gather(1, 256, pieces, nullptr));
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (int tier = 0; tier < 2; ++tier) {
            if (tier == 1) {
                ASSERT_TRUE(store.save_gather(2, 256, pieces, nullptr));  // demotes entry 1 to the host tier
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                ASSERT_TRUE(foreign());
            }
            const auto e = store.find(1);
            ASSERT_TRUE(e);
            EXPECT_EQ(e->on_host, tier == 1);
            B.set_table(d_tbl, nb, nullptr);
            ASSERT_TRUE(B.restore(b, e->data, 256, nullptr));
            EXPECT_EQ(B.position(), 256);
            EXPECT_EQ(B.match(b), 256);  // its own checkpoint from here on
            got[tier] = run_rows(B, fx.w, b, 256, {});
            ASSERT_FALSE(got[tier].empty());
        }
        want = run_rows(fresh, fx.w, b, 0, {256});
        ASSERT_GT(want.size(), got[0].size());
    }
    for (void* d : owned)
        cudaFree(d);
    for (int tier = 0; tier < 2; ++tier)
        EXPECT_EQ(std::memcmp(got[tier].data(), want.data() + (want.size() - got[tier].size()),
                              got[tier].size() * sizeof(float)),
                  0)
            << (tier ? "host tier" : "device tier");
}

// The lookahead draft moves with the snapshot: its 2 local layers' K/V of the 1024 positions before 256 go
// into the entry (kept rows flushed first) and come back on slot B, whose draft last ran another prompt and
// decoded 40 steps of it. B then drafts exactly as slot A after a rewind to its own checkpoint at 256: the
// same draft rows confirmed, bit-equal rows over the prompt tail and 120 decode steps. A draft that read
// B's own window would name other tokens.
TEST(InklingCrossSlot, RestoredDraftDraftsAsTheSlotThatRanThePrefix) {
    const char* model = std::getenv("IMP_TEST_MODEL_INKLING");
    const char* oracle = std::getenv("IMP_TEST_INKLING_ORACLE");
    if (!model || !oracle)
        GTEST_SKIP() << "IMP_TEST_MODEL_INKLING / IMP_TEST_INKLING_ORACLE not set";
    Fixture& fx = fixture();
    if (!fx.loaded)
        ASSERT_TRUE(fx.load(model));
    const std::vector<int32_t> sky = read_ids(std::string(oracle) + "/sky_prompt.json");
    const std::vector<int32_t> paris = read_ids(std::string(oracle) + "/paris_prompt.json");
    ASSERT_FALSE(sky.empty());
    ASSERT_FALSE(paris.empty());
    const int V = fx.w.cfg.vocab, H = fx.w.cfg.hidden, ctx = 4096, steps = 120;
    std::vector<int32_t> a, t, other;
    for (size_t i = 0; i < 300; ++i)
        a.push_back(sky[i % sky.size()]);
    t.assign(a.begin(), a.begin() + 280);  // b = t[0, 310), then the decode tokens
    for (size_t i = 0; i < 30 + steps; ++i)
        t.push_back(paris[i % paris.size()]);
    const std::vector<int32_t> b(t.begin(), t.begin() + 310);
    other.assign(a.rbegin(), a.rend());
    void* d_w8 = nullptr;
    float* d_sc = nullptr;
    ASSERT_EQ(cudaMalloc(&d_w8, static_cast<size_t>(V) * H), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_sc, static_cast<size_t>(V) * sizeof(float)), cudaSuccess);
    quantize_fp8_rows_async(fx.w.unembed, d_w8, V, H, d_sc, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const InklingKvLayout l = inkling_kv_apps(fx.w.cfg, true);
    const size_t KD = static_cast<size_t>(fx.w.cfg.n_kv_heads) * fx.w.cfg.head_dim;
    std::vector<float> rows[2], dec[2];
    long accepted[2] = {}, drafted[2] = {};
    std::vector<void*> owned;
    {
        Stepper st(fx.w);
        InklingDraftWeights dw;
        ASSERT_TRUE(dw.load(model, fx.va, fx.w.embed_norm, InklingHeadView{d_w8, d_sc, V}, st.s));
        InklingForward A, B;
        InklingDraft dA, dB;
        ASSERT_TRUE(A.init(fx.w, fx.va, 512, ctx, nullptr, /*own_kv=*/false));
        ASSERT_TRUE(B.init(fx.w, fx.va, 512, ctx, nullptr, /*own_kv=*/false));
        const int nb = A.table_cap();
        std::vector<half*> k, v;  // the target's 8 pool layers, then the draft's
        for (int i = 0; i < 2 * l.n_layers(); ++i) {
            void* d = nullptr;
            const size_t bytes = 2 * static_cast<size_t>(nb) * 16 * KD * sizeof(half);
            ASSERT_EQ(cudaMalloc(&d, bytes), cudaSuccess);
            ASSERT_EQ(cudaMemset(d, 0, bytes), cudaSuccess);
            owned.push_back(d);
            (i % 2 ? v : k).push_back(static_cast<half*>(d));
        }
        std::vector<int> ids(2 * static_cast<size_t>(nb));
        std::iota(ids.begin(), ids.end(), 0);
        int* d_tbl = nullptr;
        ASSERT_EQ(cudaMalloc(&d_tbl, ids.size() * sizeof(int)), cudaSuccess);
        owned.push_back(d_tbl);
        ASSERT_EQ(cudaMemcpy(d_tbl, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice),
                  cudaSuccess);
        const size_t dl = static_cast<size_t>(l.draft_layer);
        for (auto [f, d] : {std::pair{&A, &dA}, std::pair{&B, &dB}}) {
            ASSERT_TRUE(f->bind_kv(k, v));
            ASSERT_TRUE(d->init(dw, fx.va, ctx, 512, InklingForward::kSeg + InklingForward::kSlack, false));
            ASSERT_TRUE(d->bind_kv(k[dl], v[dl], f->table()));
            ASSERT_TRUE(f->enable_lookahead(d->hooks(), dw.cfg.taps, fx.w.embed));
            f->set_snapshot_draft(&dw.cfg);
        }
        // B's draft and rings hold another prompt and 40 steps of its decode, on the other pool blocks.
        B.set_table(d_tbl + nb, nb, nullptr);
        ASSERT_FALSE(run_rows(B, fx.w, other, 0, {256}).empty());
        for (int pos = 300; pos < 340; ++pos)
            ASSERT_FALSE(st.step(B, other[static_cast<size_t>(pos - 300)], pos).empty());
        A.set_table(d_tbl, nb, nullptr);
        ASSERT_FALSE(run_rows(A, fx.w, a, 0, {256}).empty());
        ASSERT_TRUE(A.snapshot_has_draft());
        ASSERT_EQ(A.snapshot_bytes(),
                  inkling_snap_layout(inkling_slab_layout(fx.w.cfg, 512), fx.w.cfg.window, &dw.cfg).bytes);
        std::vector<SnapshotPiece> pieces;
        ASSERT_TRUE(A.snapshot_pieces(256, pieces, st.s));
        RecurrentSnapshotStore store;
        store.init(A.snapshot_bytes(), A.snapshot_bytes());
        ASSERT_TRUE(store.save_gather(1, 256, pieces, st.s));
        ASSERT_EQ(cudaStreamSynchronize(st.s), cudaSuccess);
        const auto e = store.find(1);
        ASSERT_TRUE(e);
        B.set_table(d_tbl, nb, nullptr);
        EXPECT_FALSE(
            B.restore(b, e->data, 256, nullptr, /*with_draft=*/false));  // a draft slot needs its rows
        ASSERT_TRUE(B.restore(b, e->data, 256, nullptr));
        ASSERT_EQ(A.rewind(b, nullptr), 256);
        InklingForward* fw[2] = {&A, &B};
        for (int g = 0; g < 2; ++g) {
            InklingForward& f = *fw[g];
            const InklingForward::SpecStats s0 = f.spec_stats();
            rows[g] = run_rows(f, fx.w, b, 256, {});
            ASSERT_FALSE(rows[g].empty());
            for (int pos = 310; pos < 310 + steps; ++pos) {
                const std::vector<float> r = st.step(f, t[static_cast<size_t>(pos)], pos);
                ASSERT_FALSE(r.empty());
                dec[g].insert(dec[g].end(), r.begin(), r.end());
            }
            accepted[g] = f.spec_stats().accepted - s0.accepted;
            drafted[g] = f.spec_stats().drafted - s0.drafted;
        }
    }
    for (void* d : owned)
        cudaFree(d);
    cudaFree(d_w8);
    cudaFree(d_sc);
    std::printf(
        "cross-slot draft: %ld / %ld draft rows confirmed after the rewind, %ld / %ld after the restore\n",
        accepted[0], drafted[0], accepted[1], drafted[1]);
    EXPECT_GT(drafted[0], 0);
    EXPECT_EQ(drafted[1], drafted[0]);
    EXPECT_EQ(accepted[1], accepted[0]);
    EXPECT_EQ(std::memcmp(rows[0].data(), rows[1].data(), rows[0].size() * sizeof(float)), 0);
    ASSERT_EQ(dec[0].size(), dec[1].size());
    EXPECT_EQ(std::memcmp(dec[0].data(), dec[1].data(), dec[0].size() * sizeof(float)), 0);
}

}  // namespace
