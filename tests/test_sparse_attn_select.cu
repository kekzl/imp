// Sparse decode attention (attention.sparse_topk_tokens) unit tests:
//   * key min/max metadata maintenance (decode single-token, prefill spans,
//     partial-block continuation merge, block-reuse re-init, FP8 raw dequant)
//   * top-k selection: identity pass-through, forced sink/recent blocks,
//     ascending order, deterministic ties, compacted context length
//   * end-to-end: paged_attention_decode over the identity-compacted table is
//     bit-identical to the dense table
// Kernels under test: src/exec/sparse_attn_select.cu.

#include <gtest/gtest.h>
#include "exec/sparse_attn_select.h"
#include "compute/attention_paged.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <float.h>
#include <cstdint>
#include <vector>

namespace imp {
namespace {

constexpr int kBS = 16;  // kKVBlockSize

// The production entry point is the batched all-layer launcher; this wrapper
// keeps the single-layer call shape the metadata tests were written against
// (n_layers=1, zero layer strides, no ragged offsets).
static void sparse_update_key_minmax(QType t, const void* k, void* mm, const int* pos, const int* bt,
                                     int nkv, int hd, int bs, int n, int mbps, int nseq,
                                     cudaStream_t stream, const int* seq_offsets = nullptr) {
    sparse_update_key_minmax_all_layers(t, k, 0, mm, 0, pos, bt, seq_offsets, /*n_layers=*/1, nkv, hd, bs,
                                        n, mbps, nseq, stream);
}

// Deterministic value fill: distinct, sign-mixed, exactly f16-representable.
static float fill_val(int pos, int e) {
    const int v = ((pos * 131 + e * 17) % 255) - 127;
    return static_cast<float>(v) * 0.125f;
}

template <typename T>
static T* dmalloc(size_t n) {
    T* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, n * sizeof(T)), cudaSuccess);
    return p;
}

template <typename T>
static void dcopy(T* dst, const std::vector<T>& src) {
    EXPECT_EQ(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
}

template <typename T>
static std::vector<T> dread(const T* src, size_t n) {
    std::vector<T> out(n);
    EXPECT_EQ(cudaMemcpy(out.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return out;
}

// Host reference for the metadata of one block: elementwise min/max over the
// written slots' K rows.
static void ref_minmax(const std::vector<half>& k_cache, int block_id, int row_elems, int slots,
                       std::vector<float>& mn, std::vector<float>& mx) {
    mn.assign(row_elems, FLT_MAX);
    mx.assign(row_elems, -FLT_MAX);
    for (int s = 0; s < slots; s++) {
        for (int e = 0; e < row_elems; e++) {
            const float v = __half2float(
                k_cache[static_cast<size_t>(block_id) * kBS * row_elems + s * row_elems + e]);
            mn[e] = std::min(mn[e], v);
            mx[e] = std::max(mx[e], v);
        }
    }
}

class SparseMinMaxTest : public ::testing::Test {
protected:
    static constexpr int nkv = 2;
    static constexpr int hd = 64;
    static constexpr int row_elems = nkv * hd;
    static constexpr int n_blocks_pool = 8;

    void SetUp() override {
        k_cache_h.assign(static_cast<size_t>(n_blocks_pool) * kBS * row_elems, __float2half(0.0f));
        d_k = dmalloc<half>(k_cache_h.size());
        d_mm = dmalloc<__half2>(static_cast<size_t>(n_blocks_pool) * row_elems);
        // Poison the metadata: every test must overwrite what it asserts on.
        cudaMemset(d_mm, 0x7F, static_cast<size_t>(n_blocks_pool) * row_elems * sizeof(__half2));
    }
    void TearDown() override {
        cudaFree(d_k);
        cudaFree(d_mm);
    }

    void write_row(int block_id, int slot, int pos) {
        for (int e = 0; e < row_elems; e++)
            k_cache_h[static_cast<size_t>(block_id) * kBS * row_elems + slot * row_elems + e] = __float2half(
                fill_val(pos, e));
    }

    void check_block(int block_id, int slots, const char* what) {
        auto mm = dread(d_mm, static_cast<size_t>(n_blocks_pool) * row_elems);
        std::vector<float> mn, mx;
        ref_minmax(k_cache_h, block_id, row_elems, slots, mn, mx);
        for (int e = 0; e < row_elems; e++) {
            const __half2 v = mm[static_cast<size_t>(block_id) * row_elems + e];
            ASSERT_FLOAT_EQ(__low2float(v), mn[e]) << what << " min block " << block_id << " e " << e;
            ASSERT_FLOAT_EQ(__high2float(v), mx[e]) << what << " max block " << block_id << " e " << e;
        }
    }

    std::vector<half> k_cache_h;
    half* d_k = nullptr;
    __half2* d_mm = nullptr;
};

TEST_F(SparseMinMaxTest, PrefillSpanThenDecodeMerge) {
    // Prefill: 40 contiguous tokens (blocks 0,1 full; block 2 has 8 slots) in
    // one launch, flat single-seq table.
    std::vector<int> bt_h = {0, 1, 2, 3};
    int* d_bt = dmalloc<int>(bt_h.size());
    dcopy(d_bt, bt_h);
    std::vector<int> pos_h(40);
    for (int i = 0; i < 40; i++) {
        pos_h[i] = i;
        write_row(i / kBS, i % kBS, i);
    }
    int* d_pos = dmalloc<int>(pos_h.size());
    dcopy(d_pos, pos_h);
    dcopy(d_k, k_cache_h);

    sparse_update_key_minmax(QType::F16, d_k, d_mm, d_pos, d_bt, nkv, hd, kBS, 40,
                             /*max_blocks_per_seq=*/0, /*n_sequences=*/1, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    check_block(0, kBS, "prefill");
    check_block(1, kBS, "prefill");
    check_block(2, 8, "prefill partial tail");

    // Decode continuation: token at pos 40 lands in block 2 slot 8 - the
    // metadata must MERGE with the existing partial-block state.
    write_row(2, 8, 40);
    dcopy(d_k, k_cache_h);
    std::vector<int> pos2 = {40};
    int* d_pos2 = dmalloc<int>(pos2.size());
    dcopy(d_pos2, pos2);
    sparse_update_key_minmax(QType::F16, d_k, d_mm, d_pos2, d_bt, nkv, hd, kBS, 1, 0, 1, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    check_block(2, 9, "decode merge");

    // Block reuse: a fresh sequence writes pos 0 into (reused) block 3 - the
    // slot-0 write must RE-INITIALIZE, not merge with the poison pattern.
    std::vector<int> bt2_h = {3};
    int* d_bt2 = dmalloc<int>(bt2_h.size());
    dcopy(d_bt2, bt2_h);
    write_row(3, 0, 7);
    dcopy(d_k, k_cache_h);
    std::vector<int> pos3 = {0};
    int* d_pos3 = dmalloc<int>(pos3.size());
    dcopy(d_pos3, pos3);
    sparse_update_key_minmax(QType::F16, d_k, d_mm, d_pos3, d_bt2, nkv, hd, kBS, 1, 0, 1, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // Reference over block 3 slot range [0,1): equals the single row.
    {
        auto mm = dread(d_mm, static_cast<size_t>(n_blocks_pool) * row_elems);
        for (int e = 0; e < row_elems; e++) {
            const float v = fill_val(7, e);
            const __half2 got = mm[static_cast<size_t>(3) * row_elems + e];
            ASSERT_FLOAT_EQ(__low2float(got), v) << "reuse re-init min e " << e;
            ASSERT_FLOAT_EQ(__high2float(got), v) << "reuse re-init max e " << e;
        }
    }
    cudaFree(d_bt);
    cudaFree(d_bt2);
    cudaFree(d_pos);
    cudaFree(d_pos2);
    cudaFree(d_pos3);
}

TEST_F(SparseMinMaxTest, MultiSeqDecodeTwoTokens) {
    // Batched decode: 2 sequences, one token each, 2D block table.
    std::vector<int> bt_h = {4, 0, 5, 0};  // [2, mbps=2]
    int* d_bt = dmalloc<int>(bt_h.size());
    dcopy(d_bt, bt_h);
    write_row(4, 3, 100);  // seq 0 at pos 3
    write_row(4, 0, 55);   // pre-existing rows in the block (simulate earlier writes)
    write_row(4, 1, 56);
    write_row(4, 2, 57);
    write_row(5, 0, 200);  // seq 1 at pos 0 (slot 0 -> init)
    dcopy(d_k, k_cache_h);
    // Seed block 4's metadata with the first three rows (slots 0..2).
    {
        std::vector<int> pos_h = {0, 1, 2};
        int* d_pos = dmalloc<int>(pos_h.size());
        dcopy(d_pos, pos_h);
        std::vector<int> bt4 = {4};
        int* d_bt4 = dmalloc<int>(1);
        dcopy(d_bt4, bt4);
        sparse_update_key_minmax(QType::F16, d_k, d_mm, d_pos, d_bt4, nkv, hd, kBS, 3, 0, 1, nullptr);
        cudaFree(d_pos);
        cudaFree(d_bt4);
    }
    std::vector<int> pos_h = {3, 0};
    int* d_pos = dmalloc<int>(pos_h.size());
    dcopy(d_pos, pos_h);
    sparse_update_key_minmax(QType::F16, d_k, d_mm, d_pos, d_bt, nkv, hd, kBS, 2,
                             /*max_blocks_per_seq=*/2, /*n_sequences=*/2, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    check_block(4, 4, "multi-seq merge");
    check_block(5, 1, "multi-seq init");
    cudaFree(d_bt);
    cudaFree(d_pos);
}

TEST_F(SparseMinMaxTest, RepeatedPositionPadRows) {
    // Spec-chunk pad rows can repeat a position. The owner's forward span
    // must only cover CONSECUTIVE slots - a repeated position must not make
    // it read slots that were never written (slot+j walk).
    std::vector<int> bt_h = {6};
    int* d_bt = dmalloc<int>(1);
    dcopy(d_bt, bt_h);
    // Written content: slots 0..5 only. Poison the rest of the block region
    // so an over-long span read would drag garbage into the metadata.
    for (int s = 0; s <= 5; s++)
        write_row(6, s, 60 + s);
    for (int s = 6; s < kBS; s++)
        for (int e = 0; e < row_elems; e++)
            k_cache_h[static_cast<size_t>(6) * kBS * row_elems + s * row_elems + e] =
                __float2half(9999.0f);
    dcopy(d_k, k_cache_h);
    // Launch shape: pads repeat position 5 (slot 5) after the real rows 0..5.
    std::vector<int> pos_h = {0, 1, 2, 3, 4, 5, 5, 5};
    int* d_pos = dmalloc<int>(pos_h.size());
    dcopy(d_pos, pos_h);
    sparse_update_key_minmax(QType::F16, d_k, d_mm, d_pos, d_bt, nkv, hd, kBS,
                             static_cast<int>(pos_h.size()), 0, 1, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // Reference over slots 0..5 only: 9999 poison must NOT appear.
    check_block(6, 6, "repeated-position span clamp");
    cudaFree(d_bt);
    cudaFree(d_pos);
}

TEST_F(SparseMinMaxTest, RaggedSeqOffsetsMapping) {
    // Ragged prefill shape: one launch over the CONCATENATED rows of two
    // sequences; seq_offsets maps token -> block-table ROW (token_idx does
    // not equal seq). Seq 0: 20 tokens from pos 0 (blocks 0,1 of row 0);
    // seq 1: 5 tokens from pos 16 (block 3 of row 1, slot 0 -> init).
    std::vector<int> bt_h = {0, 1, /*row1:*/ 2, 3};  // [2, mbps=2]
    int* d_bt = dmalloc<int>(bt_h.size());
    dcopy(d_bt, bt_h);
    std::vector<int> pos_h(25);
    for (int i = 0; i < 20; i++) {
        pos_h[i] = i;
        write_row(i / kBS, i % kBS, i);
    }
    for (int i = 0; i < 5; i++) {
        pos_h[20 + i] = 16 + i;
        write_row(3, i, 300 + i);
    }
    int* d_pos = dmalloc<int>(pos_h.size());
    dcopy(d_pos, pos_h);
    std::vector<int> soff_h = {0, 20, 25};
    int* d_soff = dmalloc<int>(soff_h.size());
    dcopy(d_soff, soff_h);
    dcopy(d_k, k_cache_h);
    sparse_update_key_minmax(QType::F16, d_k, d_mm, d_pos, d_bt, nkv, hd, kBS, 25,
                             /*max_blocks_per_seq=*/2, /*n_sequences=*/2, nullptr, d_soff);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    check_block(0, kBS, "ragged seq0 blk0");
    check_block(1, 4, "ragged seq0 tail");
    check_block(3, 5, "ragged seq1 blk");
    cudaFree(d_bt);
    cudaFree(d_pos);
    cudaFree(d_soff);
}

TEST_F(SparseMinMaxTest, Fp8RawScaleOneDequant) {
    // FP8 cache: metadata must equal min/max of the RAW (scale-1) dequant.
    std::vector<__nv_fp8_e4m3> k8(static_cast<size_t>(n_blocks_pool) * kBS * row_elems);
    for (int s = 0; s < 5; s++)
        for (int e = 0; e < row_elems; e++)
            k8[static_cast<size_t>(0) * kBS * row_elems + s * row_elems + e] = __nv_fp8_e4m3(fill_val(s, e));
    __nv_fp8_e4m3* d_k8 = dmalloc<__nv_fp8_e4m3>(k8.size());
    dcopy(d_k8, k8);
    std::vector<int> bt_h = {0};
    int* d_bt = dmalloc<int>(1);
    dcopy(d_bt, bt_h);
    std::vector<int> pos_h = {0, 1, 2, 3, 4};
    int* d_pos = dmalloc<int>(pos_h.size());
    dcopy(d_pos, pos_h);
    sparse_update_key_minmax(QType::FP8_E4M3, d_k8, d_mm, d_pos, d_bt, nkv, hd, kBS, 5, 0, 1, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto mm = dread(d_mm, static_cast<size_t>(n_blocks_pool) * row_elems);
    for (int e = 0; e < row_elems; e++) {
        float mn = FLT_MAX, mx = -FLT_MAX;
        for (int s = 0; s < 5; s++) {
            const float v = static_cast<float>(k8[static_cast<size_t>(s) * row_elems + e]);
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        ASSERT_FLOAT_EQ(__low2float(mm[e]), mn) << "fp8 min e " << e;
        ASSERT_FLOAT_EQ(__high2float(mm[e]), mx) << "fp8 max e " << e;
    }
    cudaFree(d_k8);
    cudaFree(d_bt);
    cudaFree(d_pos);
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

class SparseSelectTest : public ::testing::Test {
protected:
    static constexpr int nh = 4;
    static constexpr int nkv = 2;
    static constexpr int hd = 64;
    static constexpr int row_elems = nkv * hd;

    // Build metadata where block b's bound for every head is `weight[b]`:
    // min = max = weight[b] on dim 0 of each kv head, 0 elsewhere, and
    // q = 1 on dim 0 of each head -> score(b) == weight[b] exactly.
    // engage_blocks defaults to budget_blocks (sparse_min_ctx off).
    void run_select(const std::vector<float>& weight, int ctx_len, int budget_blocks, int sink_blocks,
                    int recent_blocks, std::vector<int>& out_bt, int& out_ctx, int engage_blocks = 0) {
        if (engage_blocks <= 0)
            engage_blocks = budget_blocks;
        const int table_blocks = std::max(budget_blocks, engage_blocks);
        const int n_blocks = (ctx_len + kBS - 1) / kBS;
        ASSERT_LE(n_blocks, static_cast<int>(weight.size()));
        const int mbps = static_cast<int>(weight.size());

        std::vector<int> bt_h(mbps);
        for (int b = 0; b < mbps; b++)
            bt_h[b] = 100 + b;  // physical ids distinct from logical indices
        // Metadata is indexed by PHYSICAL block id.
        std::vector<__half2> mm_phys(static_cast<size_t>(mbps + 100) * row_elems,
                                     __floats2half2_rn(0.f, 0.f));
        for (int b = 0; b < mbps; b++)
            for (int kvh = 0; kvh < nkv; kvh++)
                mm_phys[static_cast<size_t>(100 + b) * row_elems + kvh * hd] = __floats2half2_rn(weight[b],
                                                                                                 weight[b]);

        std::vector<half> q_h(static_cast<size_t>(nh) * hd, __float2half(0.f));
        for (int h = 0; h < nh; h++)
            q_h[static_cast<size_t>(h) * hd] = __float2half(1.f);

        __half2* d_mm = dmalloc<__half2>(mm_phys.size());
        dcopy(d_mm, mm_phys);
        half* d_q = dmalloc<half>(q_h.size());
        dcopy(d_q, q_h);
        int* d_bt = dmalloc<int>(bt_h.size());
        dcopy(d_bt, bt_h);
        std::vector<int> ctx_h = {ctx_len};
        int* d_ctx = dmalloc<int>(1);
        dcopy(d_ctx, ctx_h);
        float* d_scores = dmalloc<float>(mbps);
        int* d_sbt = dmalloc<int>(table_blocks);
        int* d_sctx = dmalloc<int>(1);
        cudaMemset(d_sbt, 0xFF, table_blocks * sizeof(int));

        sparse_select_blocks(d_q, d_mm, d_bt, d_ctx, /*n_seq=*/1, nh, nkv, hd, kBS, mbps, budget_blocks,
                             sink_blocks, recent_blocks, engage_blocks, table_blocks, d_scores, d_sbt, d_sctx,
                             nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        out_bt = dread(d_sbt, table_blocks);
        out_ctx = dread(d_sctx, 1)[0];
        cudaFree(d_mm);
        cudaFree(d_q);
        cudaFree(d_bt);
        cudaFree(d_ctx);
        cudaFree(d_scores);
        cudaFree(d_sbt);
        cudaFree(d_sctx);
    }
};

TEST_F(SparseSelectTest, IdentityPassThrough) {
    std::vector<float> w(6, 0.f);
    std::vector<int> out;
    int ctx = 0;
    run_select(w, /*ctx_len=*/85, /*budget=*/8, /*sink=*/1, /*recent=*/1, out, ctx);
    EXPECT_EQ(ctx, 85);
    for (int b = 0; b < 6; b++)
        EXPECT_EQ(out[b], 100 + b) << "identity table row";
}

TEST_F(SparseSelectTest, EngageThresholdKeepsIdentity) {
    // 12 blocks > budget 6, but engage 16 -> identity (sparse_min_ctx regime).
    std::vector<float> w(12, 0.f);
    std::vector<int> out;
    int ctx = 0;
    run_select(w, /*ctx_len=*/190, /*budget=*/6, /*sink=*/1, /*recent=*/2, out, ctx,
               /*engage_blocks=*/16);
    EXPECT_EQ(ctx, 190);
    for (int b = 0; b < 12; b++)
        EXPECT_EQ(out[b], 100 + b) << "identity table row";
}

TEST_F(SparseSelectTest, TopKForcedAscendingDeterministic) {
    // 12 blocks (ctx 190, tail block 11 holds 14 tokens), budget 6 = 1 sink +
    // 2 recent + 3 middle picks. Middle candidates are blocks 1..9.
    // weights: block 5 and 7 clearly highest; blocks 2 and 3 TIE for third -
    // the lower index (2) must win, deterministically.
    std::vector<float> w = {0.f, 1.f, 5.f, 5.f, 0.5f, 9.f, 0.25f, 8.f, 0.75f, 1.5f, 0.f, 0.f};
    std::vector<int> out;
    int ctx = 0;
    run_select(w, /*ctx_len=*/190, /*budget=*/6, /*sink=*/1, /*recent=*/2, out, ctx);
    // Expected logical blocks: 0 (sink), 2 (tie winner), 5, 7, 10, 11 (recent).
    std::vector<int> expect = {100, 102, 105, 107, 110, 111};
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(out[i], expect[i]) << "slot " << i;
    // ctx: 5 full blocks + 14-token tail.
    EXPECT_EQ(ctx, 5 * kBS + 14);
}

TEST_F(SparseSelectTest, NegativeScoresStillSelect) {
    // All-negative bounds (uniformly negative keys): selection must still rank
    // and pick the least-negative middles, not collapse.
    std::vector<float> w = {0.f, -3.f, -1.f, -8.f, -2.f, -9.f, 0.f, 0.f};
    std::vector<int> out;
    int ctx = 0;
    run_select(w, /*ctx_len=*/8 * kBS, /*budget=*/5, /*sink=*/1, /*recent=*/2, out, ctx);
    // middles 1..5, pick 2: blocks 2 (-1) and 4 (-2). Forced: 0, 6, 7.
    std::vector<int> expect = {100, 102, 104, 106, 107};
    for (int i = 0; i < 5; i++)
        EXPECT_EQ(out[i], expect[i]) << "slot " << i;
    EXPECT_EQ(ctx, 5 * kBS);
}

TEST_F(SparseSelectTest, ChunkRowsSharedTablePerRowCtx) {
    // Spec verify chunk shape: rows are "sequences" sharing ONE replicated
    // physical table, each with its own context length. Row 0: 12 blocks
    // (selection engages), row 1: 2 blocks (pad-like, identity).
    const int mbps = 12;
    const int budget = 6, sink = 1, recent = 2;
    std::vector<float> w = {0.f, 1.f, 5.f, 4.f, 0.5f, 9.f, 0.25f, 8.f, 0.75f, 1.5f, 0.f, 0.f};
    std::vector<__half2> mm_phys(static_cast<size_t>(mbps + 100) * row_elems,
                                 __floats2half2_rn(0.f, 0.f));
    std::vector<int> bt_h(2 * mbps);
    for (int b = 0; b < mbps; b++) {
        bt_h[b] = 100 + b;
        bt_h[mbps + b] = 100 + b;  // row-replicated
        for (int kvh = 0; kvh < nkv; kvh++)
            mm_phys[static_cast<size_t>(100 + b) * row_elems + kvh * hd] =
                __floats2half2_rn(w[b], w[b]);
    }
    std::vector<half> q_h(static_cast<size_t>(2) * nh * hd, __float2half(0.f));
    for (int r = 0; r < 2; r++)
        for (int h = 0; h < nh; h++)
            q_h[(static_cast<size_t>(r) * nh + h) * hd] = __float2half(1.f);
    __half2* d_mm = dmalloc<__half2>(mm_phys.size());
    dcopy(d_mm, mm_phys);
    half* d_q = dmalloc<half>(q_h.size());
    dcopy(d_q, q_h);
    int* d_bt = dmalloc<int>(bt_h.size());
    dcopy(d_bt, bt_h);
    std::vector<int> ctx_h = {12 * kBS - 2, 2 * kBS};  // row 0 partial tail, row 1 tiny
    int* d_ctx = dmalloc<int>(2);
    dcopy(d_ctx, ctx_h);
    float* d_scores = dmalloc<float>(static_cast<size_t>(2) * mbps);
    int* d_sbt = dmalloc<int>(static_cast<size_t>(2) * budget);
    int* d_sctx = dmalloc<int>(2);
    cudaMemset(d_sbt, 0xFF, 2 * budget * sizeof(int));

    sparse_select_blocks(d_q, d_mm, d_bt, d_ctx, /*n_seq=*/2, nh, nkv, hd, kBS, mbps, budget, sink,
                         recent, /*engage=*/budget, /*table=*/budget, d_scores, d_sbt, d_sctx, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto out = dread(d_sbt, static_cast<size_t>(2) * budget);
    auto ctx = dread(d_sctx, 2);
    // Row 0: sink 0, middles top-3 of blocks 1..9 -> {5, 7, 2}, recents 10, 11.
    std::vector<int> expect0 = {100, 102, 105, 107, 110, 111};
    for (int i = 0; i < budget; i++)
        EXPECT_EQ(out[i], expect0[i]) << "row0 slot " << i;
    EXPECT_EQ(ctx[0], 5 * kBS + (kBS - 2));
    // Row 1: 2 blocks <= budget -> identity, ctx unchanged.
    EXPECT_EQ(out[budget + 0], 100);
    EXPECT_EQ(out[budget + 1], 101);
    EXPECT_EQ(ctx[1], 2 * kBS);
    cudaFree(d_mm);
    cudaFree(d_q);
    cudaFree(d_bt);
    cudaFree(d_ctx);
    cudaFree(d_scores);
    cudaFree(d_sbt);
    cudaFree(d_sctx);
}

// ---------------------------------------------------------------------------
// End-to-end: attention over an identity-compacted table is bit-identical.
// ---------------------------------------------------------------------------
TEST(SparseAttnE2E, IdentityTableBitIdentical) {
    const int nh = 4, nkv = 2, hd = 64;
    const int ctx_len = 85;  // partial tail
    const int n_blocks = (ctx_len + kBS - 1) / kBS;
    const int pool_blocks = 8;
    const size_t cache_elems = static_cast<size_t>(pool_blocks) * kBS * nkv * hd;

    std::vector<half> k_h(cache_elems, __float2half(0.f)), v_h(cache_elems, __float2half(0.f));
    for (int p = 0; p < ctx_len; p++)
        for (int e = 0; e < nkv * hd; e++) {
            k_h[static_cast<size_t>(p / kBS) * kBS * nkv * hd + (p % kBS) * nkv * hd + e] = __float2half(
                fill_val(p, e) * 0.05f);
            v_h[static_cast<size_t>(p / kBS) * kBS * nkv * hd + (p % kBS) * nkv * hd + e] = __float2half(
                fill_val(p + 7, e) * 0.05f);
        }
    std::vector<half> q_h(static_cast<size_t>(nh) * hd);
    for (size_t i = 0; i < q_h.size(); i++)
        q_h[i] = __float2half(fill_val(3, static_cast<int>(i % 191)) * 0.05f);

    half* d_k = dmalloc<half>(cache_elems);
    half* d_v = dmalloc<half>(cache_elems);
    half* d_q = dmalloc<half>(q_h.size());
    dcopy(d_k, k_h);
    dcopy(d_v, v_h);
    dcopy(d_q, q_h);
    std::vector<int> bt_h = {0, 1, 2, 3, 4, 5};
    int* d_bt = dmalloc<int>(bt_h.size());
    dcopy(d_bt, bt_h);
    std::vector<int> ctx_h = {ctx_len};
    int* d_ctx = dmalloc<int>(1);
    dcopy(d_ctx, ctx_h);

    int64_t qd[4] = {1, 1, nh, hd};
    int64_t cs[4] = {pool_blocks, kBS, nkv, hd};
    Tensor Q(d_q, QType::F16, 4, qd, true);
    Tensor K(d_k, QType::F16, 4, cs, true);
    Tensor V(d_v, QType::F16, 4, cs, true);
    half* d_o1 = dmalloc<half>(static_cast<size_t>(nh) * hd);
    half* d_o2 = dmalloc<half>(static_cast<size_t>(nh) * hd);
    Tensor O1(d_o1, QType::F16, 4, qd, true);
    Tensor O2(d_o2, QType::F16, 4, qd, true);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    paged_attention_decode(Q, K, V, O1, d_bt, d_ctx, kBS, scale, ctx_len, 0, 0.0f, nullptr, 6);

    // Identity compaction through the real selection kernel (budget >= blocks).
    __half2* d_mm = dmalloc<__half2>(static_cast<size_t>(pool_blocks) * nkv * hd);
    cudaMemset(d_mm, 0, static_cast<size_t>(pool_blocks) * nkv * hd * sizeof(__half2));
    float* d_scores = dmalloc<float>(8);
    int* d_sbt = dmalloc<int>(8);
    int* d_sctx = dmalloc<int>(1);
    sparse_select_blocks(d_q, d_mm, d_bt, d_ctx, 1, nh, nkv, hd, kBS, 6, /*budget=*/8, 1, 1,
                         /*engage=*/8, /*table=*/8, d_scores, d_sbt, d_sctx, nullptr);
    paged_attention_decode(Q, K, V, O2, d_sbt, d_sctx, kBS, scale, ctx_len, 0, 0.0f, nullptr, 8);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto o1 = dread(d_o1, static_cast<size_t>(nh) * hd);
    auto o2 = dread(d_o2, static_cast<size_t>(nh) * hd);
    for (size_t i = 0; i < o1.size(); i++)
        ASSERT_EQ(__half_as_ushort(o1[i]), __half_as_ushort(o2[i])) << "bit mismatch at " << i;

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_q);
    cudaFree(d_bt);
    cudaFree(d_ctx);
    cudaFree(d_o1);
    cudaFree(d_o2);
    cudaFree(d_mm);
    cudaFree(d_scores);
    cudaFree(d_sbt);
    cudaFree(d_sctx);
}

}  // namespace
}  // namespace imp
