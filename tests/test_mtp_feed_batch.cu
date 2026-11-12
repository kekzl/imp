// mtp_feed_batch() must match n feed-only mtp_draft_step() calls: identical mtp_pos, matching
// KV cache rows and h_final. Synthetic dense head (Qwen3.8/3.6-27B layout) keeps it model-free;
// RMS-relative tolerance since M=1 GEMV and M=n GEMM reduce in different orders. GPU required.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "compute/gemm.h"
#include "compute/mtp_forward.h"
#include "compute/mtp_forward_internal.h"
#include "core/tensor.h"
#include "model/mtp_head.h"

namespace {

constexpr int kH     = 64;   // hidden_dim
constexpr int kVocab = 128;
constexpr int kNh    = 4;    // attention heads
constexpr int kNkv   = 2;
constexpr int kHd    = 16;   // head_dim
constexpr int kDff   = 96;   // dense MLP width
constexpr int kRope  = 4;    // partial rotary (0.25 * head_dim)
constexpr int kSeq   = 64;   // MTP KV capacity
constexpr int kPairs = 33;   // deliberately not a multiple of anything

bool gpu_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Uploads host FP16 data and returns a device Tensor view. Device memory is
// intentionally leaked at process exit — this is a test binary.
imp::Tensor upload(const std::vector<__half>& h, std::initializer_list<int64_t> shape) {
    void* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, h.size() * sizeof(__half)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice),
              cudaSuccess);
    std::vector<int64_t> s(shape);
    return imp::Tensor(d, imp::QType::F16, static_cast<int>(s.size()), s.data(),
                       /*on_device=*/true);
}

std::vector<__half> random_h(std::mt19937& rng, size_t n, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<__half> v(n);
    for (auto& x : v) x = __float2half(dist(rng));
    return v;
}

struct SyntheticHead {
    imp::MtpHead head;
    imp::Tensor  tok_emb;   // [kVocab, kH] — doubles as the (unused) lm_head
    void*        d_hidden = nullptr;  // [kPairs, kH] main-model hidden rows
    std::vector<int32_t> tokens;

    void build(unsigned seed) {
        std::mt19937 rng(seed);
        const float w = 0.25f;
        head.pre_fc_norm_embedding = upload(random_h(rng, kH, w), {kH});
        head.pre_fc_norm_hidden    = upload(random_h(rng, kH, w), {kH});
        head.fc               = upload(random_h(rng, size_t{kH} * 2 * kH, w), {kH, 2 * kH});
        head.input_layernorm  = upload(random_h(rng, kH, w), {kH});
        head.post_attention_layernorm = upload(random_h(rng, kH, w), {kH});
        head.q_proj = upload(random_h(rng, size_t{2 * kNh * kHd} * kH, w), {2 * kNh * kHd, kH});
        head.k_proj = upload(random_h(rng, size_t{kNkv * kHd} * kH, w), {kNkv * kHd, kH});
        head.v_proj = upload(random_h(rng, size_t{kNkv * kHd} * kH, w), {kNkv * kHd, kH});
        head.o_proj = upload(random_h(rng, size_t{kH} * kNh * kHd, w), {kH, kNh * kHd});
        head.q_norm = upload(random_h(rng, kHd, w), {kHd});
        head.k_norm = upload(random_h(rng, kHd, w), {kHd});
        head.shared_expert_gate_proj = upload(random_h(rng, size_t{kDff} * kH, w), {kDff, kH});
        head.shared_expert_up_proj   = upload(random_h(rng, size_t{kDff} * kH, w), {kDff, kH});
        head.shared_expert_down_proj = upload(random_h(rng, size_t{kH} * kDff, w), {kH, kDff});
        head.final_norm = upload(random_h(rng, kH, w), {kH});
        head.attn_output_gate = true;
        head.attn_rope        = true;
        head.loaded           = true;

        tok_emb = upload(random_h(rng, size_t{kVocab} * kH, w), {kVocab, kH});

        auto h_rows = random_h(rng, size_t{kPairs} * kH, 1.0f);
        EXPECT_EQ(cudaMalloc(&d_hidden, h_rows.size() * sizeof(__half)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(d_hidden, h_rows.data(), h_rows.size() * sizeof(__half),
                             cudaMemcpyHostToDevice), cudaSuccess);

        std::uniform_int_distribution<int> tok(0, kVocab - 1);
        tokens.resize(kPairs);
        for (auto& t : tokens) t = tok(rng);
    }
};

void configure_ws(imp::MtpDraftWorkspace& ws) {
    ws.rope_theta   = 10000.0f;
    ws.rope_dim     = kRope;
    ws.mrope_sec0   = 1;
    ws.mrope_sec1   = 1;
    ws.mrope_sec2   = 0;  // sums to kRope / 2
    ws.rms_norm_eps = 1e-6f;
    ws.arch_norm_offset = 0.0f;
}

std::vector<float> download(const void* d, size_t n) {
    std::vector<__half> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(__half), cudaMemcpyDeviceToHost), cudaSuccess);
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = __half2float(h[i]);
    return f;
}

// RMS-relative difference between two device FP16 buffers.
float rel_rms_diff(const void* a, const void* b, size_t n) {
    auto fa = download(a, n);
    auto fb = download(b, n);
    double diff2 = 0.0, ref2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fa[i] - fb[i];
        diff2 += d * d;
        ref2  += static_cast<double>(fa[i]) * fa[i];
    }
    if (ref2 == 0.0) return diff2 == 0.0 ? 0.0f : 1.0e9f;
    return static_cast<float>(std::sqrt(diff2 / ref2));
}

class MtpFeedBatchTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!gpu_available()) GTEST_SKIP() << "no CUDA device";
    }
};

// Reference arm: kPairs feed-only per-pair steps. Batched arm: one (or two)
// mtp_feed_batch calls. The two must agree on mtp_pos, the KV cache rows and
// h_final.
TEST_F(MtpFeedBatchTest, BatchedFeedMatchesPerPairLoop) {
    SyntheticHead s;
    s.build(/*seed=*/7);

    imp::MtpDraftWorkspace ws_ref, ws_bat;
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws_ref, kH, kVocab, /*n_experts=*/0, /*top_k=*/0,
                                            /*expert_d_ff=*/0, kDff, kNh, kNkv, kHd, kSeq));
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws_bat, kH, kVocab, 0, 0, 0, kDff, kNh, kNkv, kHd,
                                            kSeq));
    ASSERT_GT(ws_bat.feed_rows_cap, 0) << "dense head must get batch scratch";
    configure_ws(ws_ref);
    configure_ws(ws_bat);

    cudaStream_t stream = nullptr;

    // Reference: per-pair feed-only loop.
    for (int j = 0; j < kPairs; ++j) {
        const void* h_j = static_cast<const char*>(s.d_hidden) +
                          static_cast<size_t>(j) * kH * sizeof(__half);
        ASSERT_TRUE(imp::mtp_draft_step(s.tokens[j], h_j, s.head, s.tok_emb, s.tok_emb,
                                        ws_ref, kH, kVocab, /*out_token_id=*/nullptr, stream))
            << "per-pair feed failed at j=" << j;
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(ws_ref.mtp_pos, kPairs);

    // Batched.
    ASSERT_TRUE(imp::mtp_feed_batch(s.tokens.data(), s.d_hidden, kPairs, s.head, s.tok_emb,
                                    ws_bat, kH, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(ws_bat.mtp_pos, kPairs);

    const size_t kv_elems = static_cast<size_t>(kPairs) * kNkv * kHd;
    EXPECT_LT(rel_rms_diff(ws_ref.d_k_cache, ws_bat.d_k_cache, kv_elems), 1e-2f);
    EXPECT_LT(rel_rms_diff(ws_ref.d_v_cache, ws_bat.d_v_cache, kv_elems), 1e-2f);
    EXPECT_LT(rel_rms_diff(ws_ref.d_h_final, ws_bat.d_h_final, kH), 2e-2f);

    imp::mtp_workspace_free(ws_ref);
    imp::mtp_workspace_free(ws_bat);
}

// Ragged multi-slot feed (batched verify): interleaved rows across three KV slots, one or two
// pairs each. Reference: single-slot batched feed per slot in order (mtp_select_slot).
// Slot KV rows and final_norm must match; an unfed slot stays untouched.
TEST_F(MtpFeedBatchTest, MultiSlotFeedMatchesPerSlotBatches) {
    SyntheticHead s;
    s.build(/*seed=*/23);
    constexpr int kSlots = 4;

    imp::MtpDraftWorkspace ws_ref, ws_ms;
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws_ref, kH, kVocab, 0, 0, 0, kDff, kNh, kNkv, kHd, kSeq, kSlots));
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws_ms, kH, kVocab, 0, 0, 0, kDff, kNh, kNkv, kHd, kSeq, kSlots));
    ASSERT_EQ(ws_ref.n_kv_slots, kSlots);
    configure_ws(ws_ref);
    configure_ws(ws_ms);
    cudaStream_t stream = nullptr;

    // Prefix per slot (positions 0..pre-1), fed the same way on both sides.
    const int pre[kSlots] = {5, 9, 0, 3};
    for (int sl = 0; sl < kSlots; ++sl) {
        if (pre[sl] == 0) continue;
        for (auto* ws : {&ws_ref, &ws_ms}) {
            imp::mtp_select_slot(*ws, sl);
            ASSERT_TRUE(imp::mtp_feed_batch(s.tokens.data(), s.d_hidden, pre[sl], s.head, s.tok_emb, *ws, kH,
                                            stream));
            ASSERT_EQ(ws->mtp_pos, pre[sl]);
        }
    }
    // The step: slot 0 gets 2 pairs, slot 1 one pair, slot 3 two pairs, slot 2 none.
    // Rows are interleaved; a slot's rows come in ascending position order.
    const int n = 5;
    const int row_slot[n] = {1, 0, 3, 0, 3};
    const int row_off[n] = {0, 0, 0, 1, 1};  // pair index within the slot's step
    int row_pos[n], row_src[n];
    int32_t row_tok[n];
    for (int r = 0; r < n; ++r) {
        row_pos[r] = pre[row_slot[r]] + row_off[r];
        row_src[r] = 12 + 2 * row_slot[r] + row_off[r];  // arbitrary hidden rows of the fixture
        row_tok[r] = s.tokens[static_cast<size_t>(row_src[r])];
    }

    // Reference: per slot, the batched single-slot feed of its rows in order.
    for (int sl : {1, 0, 3}) {
        std::vector<int32_t> toks;
        int first_src = -1, cnt = 0;
        for (int r = 0; r < n; ++r)
            if (row_slot[r] == sl) {
                if (first_src < 0) first_src = row_src[r];
                toks.push_back(row_tok[r]);
                ++cnt;
            }
        imp::mtp_select_slot(ws_ref, sl);
        const void* rows = static_cast<const char*>(s.d_hidden) + static_cast<size_t>(first_src) * kH * sizeof(__half);
        ASSERT_TRUE(imp::mtp_feed_batch(toks.data(), rows, cnt, s.head, s.tok_emb, ws_ref, kH, stream));
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    imp::mtp_select_slot(ws_ref, 0);

    // Multi-slot: one launch. The row order above is not slot-major on purpose.
    ASSERT_TRUE(imp::mtp_feed_rows_multislot(row_tok, s.d_hidden, row_src, n, row_slot, row_pos, s.head, s.tok_emb,
                                             ws_ms, kH, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    const size_t row_elems = static_cast<size_t>(kNkv) * kHd;
    for (int sl = 0; sl < kSlots; ++sl) {
        int fed = 0;
        for (int r = 0; r < n; ++r) fed += row_slot[r] == sl;
        const size_t elems = static_cast<size_t>(pre[sl] + fed) * row_elems;
        if (elems == 0) continue;
        const __half* kr = static_cast<const __half*>(ws_ref.d_k_cache_base) + static_cast<size_t>(sl) * ws_ref.kv_slot_elems;
        const __half* km = static_cast<const __half*>(ws_ms.d_k_cache_base) + static_cast<size_t>(sl) * ws_ms.kv_slot_elems;
        const __half* vr = static_cast<const __half*>(ws_ref.d_v_cache_base) + static_cast<size_t>(sl) * ws_ref.kv_slot_elems;
        const __half* vm = static_cast<const __half*>(ws_ms.d_v_cache_base) + static_cast<size_t>(sl) * ws_ms.kv_slot_elems;
        EXPECT_LT(rel_rms_diff(kr, km, elems), 1e-2f) << "slot " << sl << " K";
        EXPECT_LT(rel_rms_diff(vr, vm, elems), 1e-2f) << "slot " << sl << " V";
    }
    // final_norm: the reference keeps only its LAST feed's last row (slot 3)
    // in d_h_final; the multi-slot call has every row in d_b_h_final.
    {
        int last = -1;
        for (int r = 0; r < n; ++r) if (row_slot[r] == 3) last = r;
        const __half* hm = static_cast<const __half*>(ws_ms.d_b_h_final) + static_cast<size_t>(last) * kH;
        EXPECT_LT(rel_rms_diff(ws_ref.d_h_final, hm, kH), 2e-2f) << "final_norm of the last slot-3 row";
    }
    imp::mtp_workspace_free(ws_ref);
    imp::mtp_workspace_free(ws_ms);
}

// The batched head pass: its 7 packed tables in one H2D, then the device-table feed and the chain feed.
// Same kernels and inputs as the host-table feeds: KV slots and final_norm rows byte-identical.
TEST_F(MtpFeedBatchTest, PackedPassTablesMatchHostTablesBytewise) {
    SyntheticHead s;
    s.build(/*seed=*/29);
    constexpr int kSlots = 3;
    imp::MtpDraftWorkspace ws_host, ws_pass;
    for (auto* ws : {&ws_host, &ws_pass}) {
        ASSERT_TRUE(
            imp::mtp_workspace_allocate(*ws, kH, kVocab, 0, 0, 0, kDff, kNh, kNkv, kHd, kSeq, kSlots));
        configure_ws(*ws);
    }
    cudaStream_t stream = nullptr;
    const int n = 5, m = 3;
    const int slot[n] = {2, 0, 0, 1, 2};
    const int pos[n] = {0, 0, 1, 0, 1};
    const int src[n] = {7, 3, 4, 9, 8};
    int32_t tok[n];
    for (int r = 0; r < n; ++r)
        tok[r] = s.tokens[static_cast<size_t>(src[r])];
    const int last[m] = {2, 3, 4};  // last fed row of slots 0, 1, 2
    const int cslot[m] = {0, 1, 2};
    const int cpos[m] = {2, 1, 2};
    const int32_t ctok[m] = {5, 17, 99};
    {
        int32_t* d_ctok = nullptr;
        ASSERT_EQ(cudaMalloc(&d_ctok, sizeof(ctok)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_ctok, ctok, sizeof(ctok), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_TRUE(imp::mtp_feed_rows_multislot(tok, s.d_hidden, src, n, slot, pos, s.head, s.tok_emb,
                                                 ws_host, kH, stream));
        ASSERT_TRUE(imp::mtp_feed_rows_multislot(d_ctok, ws_host.d_b_h_final, last, m, cslot, cpos, s.head,
                                                 s.tok_emb, ws_host, kH, stream));
        imp::MtpPassTables tab;
        ASSERT_TRUE(
            imp::mtp_head_pass_upload(ws_pass, tok, src, slot, pos, n, last, cslot, cpos, m, stream, tab));
        EXPECT_EQ(tab.max_pos, 1);
        EXPECT_EQ(tab.chain_max_pos, 2);
        ASSERT_TRUE(imp::mtp_feed_rows_multislot_dev(tab.tokens, s.d_hidden, tab.src, n, tab.slots, tab.pos,
                                                     tab.max_pos, s.head, s.tok_emb, ws_pass, kH, stream));
        ASSERT_TRUE(imp::mtp_feed_rows_multislot_dev(d_ctok, ws_pass.d_b_h_final, tab.last, m,
                                                     tab.chain_slots, tab.chain_pos[0], tab.chain_max_pos,
                                                     s.head, s.tok_emb, ws_pass, kH, stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        int back[m];
        ASSERT_EQ(cudaMemcpy(back, tab.chain_pos[0], sizeof(back), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_TRUE(std::equal(back, back + m, cpos));
        cudaFree(d_ctok);
    }
    auto same = [](const void* a, const void* b, size_t nb) {
        std::vector<char> ha(nb), hb(nb);
        EXPECT_EQ(cudaMemcpy(ha.data(), a, nb, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(hb.data(), b, nb, cudaMemcpyDeviceToHost), cudaSuccess);
        return ha == hb;
    };
    // Written rows only (fresh cudaMalloc memory is not zeroed): feed + chain rows per slot.
    const int written[kSlots] = {3, 2, 3};
    const size_t row_bytes = static_cast<size_t>(kNkv) * kHd * sizeof(__half);
    for (int sl = 0; sl < kSlots; ++sl) {
        const size_t off = static_cast<size_t>(sl) * ws_host.kv_slot_elems * sizeof(__half);
        const size_t nb = static_cast<size_t>(written[sl]) * row_bytes;
        EXPECT_TRUE(same(static_cast<char*>(ws_host.d_k_cache_base) + off,
                         static_cast<char*>(ws_pass.d_k_cache_base) + off, nb))
            << "K slot " << sl;
        EXPECT_TRUE(same(static_cast<char*>(ws_host.d_v_cache_base) + off,
                         static_cast<char*>(ws_pass.d_v_cache_base) + off, nb))
            << "V slot " << sl;
    }
    EXPECT_TRUE(same(ws_host.d_b_h_final, ws_pass.d_b_h_final, static_cast<size_t>(m) * kH * sizeof(__half)))
        << "chain final_norm rows";
    // m > n, or a row past the slot count: refused before touching the staging.
    imp::MtpPassTables tab;
    EXPECT_FALSE(
        imp::mtp_head_pass_upload(ws_pass, tok, src, slot, pos, 2, last, cslot, cpos, 3, stream, tab));
    const int bad_slot[n] = {2, 0, 0, kSlots, 2};
    EXPECT_FALSE(
        imp::mtp_head_pass_upload(ws_pass, tok, src, bad_slot, pos, n, last, cslot, cpos, m, stream, tab));
    imp::mtp_workspace_free(ws_host);
    imp::mtp_workspace_free(ws_pass);
}

// The head pass as a captured graph (speculative.mtp_head_graph): feed + chain captured with 1-row scan
// smem, replayed after mtp_scan_nodes_set_rows widened both scan nodes to the pass's largest position (2002:
// 8 KiB of scores). KV slots and final_norm rows bytewise equal to the eager pass; 1-row smem left in place
// traps in the scan kernel.
TEST_F(MtpFeedBatchTest, CapturedHeadPassMatchesEagerBytewise) {
    SyntheticHead s;
    s.build(/*seed=*/31);
    constexpr int kSlots = 2;
    constexpr int kSeqLong = 4096;
    imp::MtpDraftWorkspace ws_e, ws_g;
    for (auto* ws : {&ws_e, &ws_g}) {
        ASSERT_TRUE(
            imp::mtp_workspace_allocate(*ws, kH, kVocab, 0, 0, 0, kDff, kNh, kNkv, kHd, kSeqLong, kSlots));
        configure_ws(*ws);
        // Zero prefix rows on both sides: the scans read every row below each position.
        const size_t kv_bytes = static_cast<size_t>(kSlots) * ws->kv_slot_elems * sizeof(__half);
        ASSERT_EQ(cudaMemset(ws->d_k_cache_base, 0, kv_bytes), cudaSuccess);
        ASSERT_EQ(cudaMemset(ws->d_v_cache_base, 0, kv_bytes), cudaSuccess);
    }
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    const int n = 3, m = 2;
    const int slot[n] = {0, 0, 1};
    const int pos[n] = {2000, 2001, 700};
    const int src[n] = {20, 21, 25};
    int32_t tok[n];
    for (int r = 0; r < n; ++r)
        tok[r] = s.tokens[static_cast<size_t>(src[r])];
    const int last[m] = {1, 2};
    const int cslot[m] = {0, 1};
    const int cpos[m] = {2002, 701};
    const int32_t ctok[m] = {7, 101};
    int32_t* d_ctok = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ctok, sizeof(ctok)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_ctok, ctok, sizeof(ctok), cudaMemcpyHostToDevice), cudaSuccess);
    auto pass = [&](imp::MtpDraftWorkspace& ws, const imp::MtpPassTables& tab, int feed_max, int chain_max) {
        return imp::mtp_feed_rows_multislot_dev(tab.tokens, s.d_hidden, tab.src, n, tab.slots, tab.pos,
                                                feed_max, s.head, s.tok_emb, ws, kH, stream) &&
               imp::mtp_feed_rows_multislot_dev(d_ctok, ws.d_b_h_final, tab.last, m, tab.chain_slots,
                                                tab.chain_pos[0], chain_max, s.head, s.tok_emb, ws, kH,
                                                stream);
    };
    imp::MtpPassTables tab_e, tab_g;
    ASSERT_TRUE(imp::mtp_head_pass_upload(ws_e, tok, src, slot, pos, n, last, cslot, cpos, m, stream, tab_e));
    ASSERT_TRUE(pass(ws_e, tab_e, tab_e.max_pos, tab_e.chain_max_pos));  // also warms cuBLASLt's shapes
    ASSERT_TRUE(imp::mtp_head_pass_upload(ws_g, tok, src, slot, pos, n, last, cslot, cpos, m, stream, tab_g));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    imp::gemm_set_lt_capture_allowed(true);
    const bool captured = pass(ws_g, tab_g, 0, 0);
    imp::gemm_set_lt_capture_allowed(false);
    cudaGraph_t graph = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    ASSERT_TRUE(captured);
    imp::MtpScanNodes scan;
    ASSERT_TRUE(imp::mtp_scan_nodes_find(graph, scan));
    ASSERT_EQ(scan.nodes.size(), 2u) << "feed scan + chain scan";
    EXPECT_EQ(scan.params[0].sharedMemBytes, imp::mtp_scan_node_smem(scan.params[0].func, 1));
    EXPECT_GT(imp::mtp_attn_scan_group_of(scan.params[0].func), 0) << "3 feed rows take the GQA-grouped scan";
    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
    const int rows = imp::mtp_scan_smem_rows(std::max(tab_g.max_pos, tab_g.chain_max_pos), kSeqLong);
    EXPECT_EQ(rows, 2048);
    ASSERT_TRUE(imp::mtp_scan_nodes_set_rows(exec, scan, rows));
    ASSERT_EQ(cudaGraphLaunch(exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    auto same = [](const void* a, const void* b, size_t nb) {
        std::vector<char> ha(nb), hb(nb);
        EXPECT_EQ(cudaMemcpy(ha.data(), a, nb, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(hb.data(), b, nb, cudaMemcpyDeviceToHost), cudaSuccess);
        return ha == hb;
    };
    // Whole caches: zero prefix plus the feed and chain rows of both slots.
    const size_t kv_bytes = static_cast<size_t>(kSlots) * ws_e.kv_slot_elems * sizeof(__half);
    EXPECT_TRUE(same(ws_e.d_k_cache_base, ws_g.d_k_cache_base, kv_bytes)) << "K";
    EXPECT_TRUE(same(ws_e.d_v_cache_base, ws_g.d_v_cache_base, kv_bytes)) << "V";
    EXPECT_TRUE(same(ws_e.d_b_h_final, ws_g.d_b_h_final, static_cast<size_t>(m) * kH * sizeof(__half)))
        << "chain final_norm rows";
    EXPECT_EQ(cudaGraphExecDestroy(exec), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    cudaFree(d_ctok);
    cudaStreamDestroy(stream);
    imp::mtp_workspace_free(ws_e);
    imp::mtp_workspace_free(ws_g);
}

// A second batch call must continue at the right cache position and RoPE
// phase: 10 + 23 batched pairs vs 33 per-pair. A base-offset bug (rows
// rotated or appended from position 0 again) fails this immediately.
TEST_F(MtpFeedBatchTest, SecondBatchContinuesAtCorrectPosition) {
    SyntheticHead s;
    s.build(/*seed=*/11);

    imp::MtpDraftWorkspace ws_ref, ws_bat;
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws_ref, kH, kVocab, 0, 0, 0, kDff, kNh, kNkv, kHd,
                                            kSeq));
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws_bat, kH, kVocab, 0, 0, 0, kDff, kNh, kNkv, kHd,
                                            kSeq));
    configure_ws(ws_ref);
    configure_ws(ws_bat);
    cudaStream_t stream = nullptr;

    for (int j = 0; j < kPairs; ++j) {
        const void* h_j = static_cast<const char*>(s.d_hidden) +
                          static_cast<size_t>(j) * kH * sizeof(__half);
        ASSERT_TRUE(imp::mtp_draft_step(s.tokens[j], h_j, s.head, s.tok_emb, s.tok_emb,
                                        ws_ref, kH, kVocab, nullptr, stream));
    }

    constexpr int kFirst = 10;
    ASSERT_TRUE(imp::mtp_feed_batch(s.tokens.data(), s.d_hidden, kFirst, s.head, s.tok_emb,
                                    ws_bat, kH, stream));
    ASSERT_EQ(ws_bat.mtp_pos, kFirst);
    const void* rest = static_cast<const char*>(s.d_hidden) +
                       static_cast<size_t>(kFirst) * kH * sizeof(__half);
    ASSERT_TRUE(imp::mtp_feed_batch(s.tokens.data() + kFirst, rest, kPairs - kFirst, s.head,
                                    s.tok_emb, ws_bat, kH, stream));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(ws_bat.mtp_pos, kPairs);

    const size_t kv_elems = static_cast<size_t>(kPairs) * kNkv * kHd;
    EXPECT_LT(rel_rms_diff(ws_ref.d_k_cache, ws_bat.d_k_cache, kv_elems), 1e-2f);
    EXPECT_LT(rel_rms_diff(ws_ref.d_v_cache, ws_bat.d_v_cache, kv_elems), 1e-2f);
    EXPECT_LT(rel_rms_diff(ws_ref.d_h_final, ws_bat.d_h_final, kH), 2e-2f);

    imp::mtp_workspace_free(ws_ref);
    imp::mtp_workspace_free(ws_bat);
}

// Guardrails: a MoE-shaped workspace must report no batch capability; the head KV is a ring of kSeq rows:
// a feed past its end wraps (and raises the slot's d_slot_hi), a feed wider than the ring refuses.
TEST_F(MtpFeedBatchTest, RefusesUnsupportedOrOverCapacity) {
    SyntheticHead s;
    s.build(/*seed=*/13);

    imp::MtpDraftWorkspace ws;
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws, kH, kVocab, 0, 0, 0, kDff, kNh, kNkv, kHd,
                                            kSeq));
    configure_ws(ws);
    ws.mtp_pos = kSeq - 4;  // 4 rows to the ring end
    EXPECT_TRUE(imp::mtp_feed_batch(s.tokens.data(), s.d_hidden, 8, s.head, s.tok_emb, ws, kH, nullptr));
    EXPECT_EQ(ws.mtp_pos, kSeq + 4);
    int hi = 0;
    ASSERT_EQ(cudaMemcpy(&hi, ws.d_slot_hi, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(hi, kSeq + 4);
    if (ws.feed_rows_cap > kSeq) {
        EXPECT_FALSE(
            imp::mtp_feed_batch(s.tokens.data(), s.d_hidden, kSeq + 1, s.head, s.tok_emb, ws, kH, nullptr));
        EXPECT_EQ(ws.mtp_pos, kSeq + 4) << "failed feed must not advance the cache";
    }
    ASSERT_TRUE(imp::mtp_kv_hi_reset(ws, -1, nullptr));
    ASSERT_EQ(cudaMemcpy(&hi, ws.d_slot_hi, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(hi, 0);
    imp::mtp_workspace_free(ws);

    // MoE-shaped alloc (n_experts > 0) gets no batch scratch.
    imp::MtpDraftWorkspace ws_moe;
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws_moe, kH, kVocab, /*n_experts=*/4, /*top_k=*/2,
                                            /*expert_d_ff=*/32, kDff, kNh, kNkv, kHd, kSeq));
    EXPECT_EQ(ws_moe.feed_rows_cap, 0);
    imp::mtp_workspace_free(ws_moe);
}

// The capacity plan prices the workspace with mtp_workspace_bytes (CPU, no allocator); the bytes
// the allocator actually requests must equal it, dense (feed + carry + prenorm) and MoE.
TEST_F(MtpFeedBatchTest, AllocatedBytesEqualThePlanMirror) {
    imp::MtpWorkspaceShape d;
    d.hidden_dim = kH;
    d.vocab_size = kVocab;
    d.shared_d_ff = kDff;
    d.num_heads = kNh;
    d.num_kv_heads = kNkv;
    d.head_dim = kHd;
    d.max_seq_len = kSeq;
    d.n_kv_slots = 4;
    d.carry_rows = 16;
    d.prenorm_rows = 64;
    imp::MtpDraftWorkspace ws;
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws, d.hidden_dim, d.vocab_size, 0, 0, 0, d.shared_d_ff,
                                            d.num_heads, d.num_kv_heads, d.head_dim, d.max_seq_len,
                                            d.n_kv_slots));
    ASSERT_TRUE(imp::mtp_carry_allocate(ws, d.carry_rows));
    ASSERT_TRUE(imp::mtp_prenorm_allocate(ws, d.prenorm_rows));
    EXPECT_EQ(ws.device_bytes, imp::mtp_workspace_bytes(d));
    imp::mtp_workspace_free(ws);
    EXPECT_EQ(ws.device_bytes, 0u);

    imp::MtpWorkspaceShape m = d;
    m.n_experts = 4;
    m.top_k = 2;
    m.expert_d_ff = 32;
    m.n_kv_slots = 1;
    m.carry_rows = 0;
    m.prenorm_rows = 0;
    imp::MtpDraftWorkspace ws_moe;
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws_moe, m.hidden_dim, m.vocab_size, m.n_experts, m.top_k,
                                            m.expert_d_ff, m.shared_d_ff, m.num_heads, m.num_kv_heads,
                                            m.head_dim, m.max_seq_len));
    EXPECT_EQ(ws_moe.device_bytes, imp::mtp_workspace_bytes(m));
    imp::mtp_workspace_free(ws_moe);
}
// mtp_attn_scan vs an FP64 softmax reference: Qwen3.8 shape (24/4 heads, head_dim 256) and a
// 128-dim one, 1 row (direct and split-K) and 3 rows, contexts 1..20003 rows (warp, pass, split tails, split
// cap).
TEST_F(MtpFeedBatchTest, AttnScanMatchesSoftmaxReference) {
    struct Shape {
        int nh, nkv, hd;
    };
    for (const Shape s : {Shape{24, 4, 256}, Shape{8, 2, 128}}) {
        for (const int base : {0, 36, 699, 4096, 20000}) {
            for (const int mode : {0, 1, 2}) {  // 1 row, 1 row split-K, 3 rows
                const int rows = mode == 2 ? 3 : 1;
                std::mt19937 rng(static_cast<unsigned>((base * 7) + rows + s.hd));
                const int ctx = base + rows;
                const auto q = random_h(rng, static_cast<size_t>(rows) * s.nh * s.hd, 1.0f);
                const auto k = random_h(rng, static_cast<size_t>(ctx) * s.nkv * s.hd, 1.0f);
                const auto v = random_h(rng, static_cast<size_t>(ctx) * s.nkv * s.hd, 1.0f);
                const imp::Tensor dq = upload(q, {rows, s.nh * s.hd});
                const imp::Tensor dk = upload(k, {ctx, s.nkv * s.hd});
                const imp::Tensor dv = upload(v, {ctx, s.nkv * s.hd});
                std::vector<__half> out(q.size());
                const imp::Tensor dout = upload(out, {rows, s.nh * s.hd});
                const size_t n_part = static_cast<size_t>(s.nh) * 32 * (s.hd + 2);  // kMtpScanMaxSplits
                float* part = nullptr;
                ASSERT_EQ(cudaMalloc(&part, n_part * sizeof(float)), cudaSuccess);
                imp::mtp_attn_scan(dq.data, dk.data, dv.data, dout.data, base, rows, s.nh, s.nkv, s.hd,
                                   nullptr, mode == 1 ? part : nullptr);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(out.data(), dout.data, out.size() * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
                double diff2 = 0.0, ref2 = 0.0;
                for (int r = 0; r < rows; ++r) {
                    for (int h = 0; h < s.nh; ++h) {
                        const int kv_h = h / (s.nh / s.nkv);
                        const int seq = base + r + 1;
                        std::vector<double> p(static_cast<size_t>(seq));
                        double mx = -1e300;
                        for (int t = 0; t < seq; ++t) {
                            double dot = 0.0;
                            for (int d = 0; d < s.hd; ++d)
                                dot += static_cast<double>(__half2float(q[((r * s.nh + h) * s.hd) + d])) *
                                       __half2float(k[((static_cast<size_t>(t) * s.nkv + kv_h) * s.hd) + d]);
                            p[t] = dot / std::sqrt(static_cast<double>(s.hd));
                            mx = std::max(mx, p[t]);
                        }
                        double sum = 0.0;
                        for (double& x : p)
                            sum += (x = std::exp(x - mx));
                        for (int d = 0; d < s.hd; ++d) {
                            double ref = 0.0;
                            for (int t = 0; t < seq; ++t)
                                ref += p[t] / sum *
                                       __half2float(v[((static_cast<size_t>(t) * s.nkv + kv_h) * s.hd) + d]);
                            const double got = __half2float(out[((r * s.nh + h) * s.hd) + d]);
                            diff2 += (got - ref) * (got - ref);
                            ref2 += ref * ref;
                        }
                    }
                }
                EXPECT_LT(std::sqrt(diff2 / ref2), 2e-3)
                    << "nh=" << s.nh << " hd=" << s.hd << " base=" << base << " mode=" << mode;
                ASSERT_EQ(cudaFree(part), cudaSuccess);
            }
        }
    }
}

// The unrolled score loop stops at the window: a 48-row window (warp 0's 4-row group ends at row 47) with a
// row 48 whose score would underflow every real row's exp. FP64 reference over rows [0, 48).
TEST_F(MtpFeedBatchTest, AttnScanIgnoresTheRowPastTheWindow) {
    const int nh = 8, nkv = 2, hd = 128, seq = 48;
    const size_t row = static_cast<size_t>(nkv) * hd;
    std::mt19937 rng(48);
    const std::vector<__half> q(static_cast<size_t>(nh) * hd, __float2half(1.0f));
    auto k = random_h(rng, (seq + 1) * row, 0.1f);
    std::fill(k.begin() + static_cast<std::ptrdiff_t>(seq * row), k.end(), __float2half(16.0f));
    const auto v = random_h(rng, (seq + 1) * row, 1.0f);
    std::vector<__half> out(q.size());
    const imp::Tensor dq = upload(q, {1, nh * hd});
    const imp::Tensor dk = upload(k, {seq + 1, static_cast<int64_t>(row)});
    const imp::Tensor dv = upload(v, {seq + 1, static_cast<int64_t>(row)});
    const imp::Tensor dout = upload(out, {1, nh * hd});
    imp::mtp_attn_scan(dq.data, dk.data, dv.data, dout.data, seq - 1, 1, nh, nkv, hd, nullptr, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(out.data(), dout.data, out.size() * sizeof(__half), cudaMemcpyDeviceToHost),
              cudaSuccess);
    double diff2 = 0.0, ref2 = 0.0;
    for (int h = 0; h < nh; ++h) {
        const int kv_h = h / (nh / nkv);
        std::vector<double> p(seq);
        double mx = -1e300, sum = 0.0;
        for (int t = 0; t < seq; ++t) {
            double dot = 0.0;
            for (int d = 0; d < hd; ++d)
                dot += __half2float(k[(t * row) + (kv_h * hd) + d]);
            p[t] = dot / std::sqrt(static_cast<double>(hd));
            mx = std::max(mx, p[t]);
        }
        for (double& x : p)
            sum += (x = std::exp(x - mx));
        for (int d = 0; d < hd; ++d) {
            double ref = 0.0;
            for (int t = 0; t < seq; ++t)
                ref += p[t] / sum * __half2float(v[(t * row) + (kv_h * hd) + d]);
            const double got = __half2float(out[(h * hd) + d]);
            diff2 += (got - ref) * (got - ref);
            ref2 += ref * ref;
        }
    }
    EXPECT_LT(std::sqrt(diff2 / ref2), 2e-3);
}

// Head KV ring (speculative.mtp_kv_window): position t lives in row t % ring. The query at p attends
// [max(p - ring + 1, hi - ring), p] (hi - 1 = a stale draft past p); FP64 reference over those positions, for
// the direct, split-K and multi-slot scans, before the wrap (lo 0) and after it (lo > 0, window wraps the
// ring end).
TEST_F(MtpFeedBatchTest, AttnScanRingWindowMatchesReference) {
    const int nh = 8, nkv = 2, hd = 128, ring = 1024;
    const size_t row = static_cast<size_t>(nkv) * hd;
    for (const int p : {700, 3000, 5000}) {
        for (const int stale : {0, 5, 300, 900}) {
            const int hi = p + 1 + stale;
            const int lo = imp::mtp_ring_lo(p, hi, ring);
            std::mt19937 rng(static_cast<unsigned>(p + stale));
            const auto q = random_h(rng, static_cast<size_t>(nh) * hd, 1.0f);
            const auto k_pos = random_h(rng, static_cast<size_t>(hi) * row, 1.0f);  // K of positions [0, hi)
            const auto v_pos = random_h(rng, static_cast<size_t>(hi) * row, 1.0f);
            // Two slots, the query's in slot 1: rows written in position order, the last write wins.
            std::vector<__half> k_ring(2 * ring * row, __float2half(0.0f)), v_ring = k_ring;
            for (int t = 0; t < hi; ++t)
                for (size_t e = 0; e < row; ++e) {
                    const size_t dst = ((static_cast<size_t>(ring) + (t % ring)) * row) + e;
                    k_ring[dst] = k_pos[(static_cast<size_t>(t) * row) + e];
                    v_ring[dst] = v_pos[(static_cast<size_t>(t) * row) + e];
                }
            std::vector<double> ref(static_cast<size_t>(nh) * hd);
            for (int h = 0; h < nh; ++h) {
                const int kv_h = h / (nh / nkv);
                std::vector<double> s(static_cast<size_t>(p + 1 - lo));
                double mx = -1e300, sum = 0.0;
                for (int t = lo; t <= p; ++t) {
                    double dot = 0.0;
                    for (int d = 0; d < hd; ++d)
                        dot += static_cast<double>(__half2float(q[(h * hd) + d])) *
                               __half2float(k_pos[(static_cast<size_t>(t) * row) + (kv_h * hd) + d]);
                    s[t - lo] = dot / std::sqrt(static_cast<double>(hd));
                    mx = std::max(mx, s[t - lo]);
                }
                for (double& x : s)
                    sum += (x = std::exp(x - mx));
                for (int t = lo; t <= p; ++t)
                    for (int d = 0; d < hd; ++d)
                        ref[(h * hd) + d] += s[t - lo] / sum *
                                             __half2float(
                                                 v_pos[(static_cast<size_t>(t) * row) + (kv_h * hd) + d]);
            }
            const imp::Tensor dq = upload(q, {1, nh * hd});
            const imp::Tensor dk = upload(k_ring, {2 * ring, static_cast<int64_t>(row)});
            const imp::Tensor dv = upload(v_ring, {2 * ring, static_cast<int64_t>(row)});
            const std::vector<int> host_tab = {0, hi, 1,
                                               p};  // d_hi [2], then the slots row's slot and position
            int* d_hi = nullptr;
            ASSERT_EQ(cudaMalloc(&d_hi, host_tab.size() * sizeof(int)), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(d_hi, host_tab.data(), host_tab.size() * sizeof(int),
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
            float* part = nullptr;
            ASSERT_EQ(cudaMalloc(&part, static_cast<size_t>(nh) * 32 * (hd + 2) * sizeof(float)),
                      cudaSuccess);
            const auto* k1 = static_cast<const __half*>(dk.data) + (ring * row);
            const auto* v1 = static_cast<const __half*>(dv.data) + (ring * row);
            for (const int mode : {0, 1, 2, 3}) {  // direct, split-K, multi-slot, multi-slot split-K
                std::vector<__half> out(q.size());
                const imp::Tensor dout = upload(out, {1, nh * hd});
                if (mode >= 2)
                    imp::mtp_attn_scan_slots(dq.data, dk.data, dv.data, dout.data, d_hi + 2, d_hi + 3,
                                             static_cast<int64_t>(ring) * row, p, 1, nh, nkv, hd, nullptr,
                                             ring, d_hi, mode == 3 ? part : nullptr);
                else
                    imp::mtp_attn_scan(dq.data, k1, v1, dout.data, p, 1, nh, nkv, hd, nullptr,
                                       mode == 1 ? part : nullptr, ring, d_hi + 1);
                ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
                ASSERT_EQ(cudaMemcpy(out.data(), dout.data, out.size() * sizeof(__half),
                                     cudaMemcpyDeviceToHost),
                          cudaSuccess);
                double diff2 = 0.0, ref2 = 0.0;
                for (size_t i = 0; i < out.size(); ++i) {
                    const double got = __half2float(out[i]);
                    diff2 += (got - ref[i]) * (got - ref[i]);
                    ref2 += ref[i] * ref[i];
                }
                EXPECT_LT(std::sqrt(diff2 / ref2), 2e-3)
                    << "p=" << p << " stale=" << stale << " mode=" << mode;
            }
            ASSERT_EQ(cudaFree(part), cudaSuccess);
            ASSERT_EQ(cudaFree(d_hi), cudaSuccess);
        }
    }
}

// Multi-slot split-K (kMtpScanSlotSplits CTAs per row and head) vs the direct multi-slot scan: Qwen3.8 head
// shape, 1 / 3 / kMtpScanSlotSplitRows rows over two ring slots, windows 1 (fewer rows than splits), 37, 1024
// (full ring) and wrapped; relative RMS < 1e-3 (summation order only).
TEST_F(MtpFeedBatchTest, AttnScanSlotsSplitMatchesDirect) {
    const int nh = 24, nkv = 4, hd = 256, ring = 1024;
    const int64_t slot_elems = static_cast<int64_t>(ring) * nkv * hd;
    std::mt19937 rng(77);
    const auto kv = random_h(rng, static_cast<size_t>(2 * slot_elems), 1.0f);
    const imp::Tensor dk = upload(kv, {2 * ring, nkv * hd});
    const auto kv2 = random_h(rng, static_cast<size_t>(2 * slot_elems), 1.0f);
    const imp::Tensor dv = upload(kv2, {2 * ring, nkv * hd});
    float* part = nullptr;
    ASSERT_EQ(cudaMalloc(&part, imp::mtp_scan_part_bytes(nh, hd)), cudaSuccess);
    for (const int rows : {1, 3, imp::kMtpScanSlotSplitRows}) {
        std::vector<int> tab(2 + (2 * static_cast<size_t>(rows)));  // d_hi [2], slots [rows], pos [rows]
        const int pos_of[4] = {0, 36, 1023, 2900};
        int max_pos = 0;
        for (int r = 0; r < rows; ++r) {
            tab[2 + r] = r % 2;
            tab[2 + rows + r] = pos_of[r % 4] + (r / 4);
            max_pos = std::max(max_pos, tab[2 + rows + r]);
        }
        tab[0] = tab[1] = max_pos + 1;
        int* d_tab = nullptr;
        ASSERT_EQ(cudaMalloc(&d_tab, tab.size() * sizeof(int)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_tab, tab.data(), tab.size() * sizeof(int), cudaMemcpyHostToDevice),
                  cudaSuccess);
        const auto q = random_h(rng, static_cast<size_t>(rows) * nh * hd, 1.0f);
        const imp::Tensor dq = upload(q, {rows, nh * hd});
        std::vector<__half> direct(q.size()), split(q.size());
        for (const bool use_split : {false, true}) {
            const imp::Tensor dout = upload(direct, {rows, nh * hd});
            imp::mtp_attn_scan_slots(dq.data, dk.data, dv.data, dout.data, d_tab + 2, d_tab + 2 + rows,
                                     slot_elems, max_pos, rows, nh, nkv, hd, nullptr, ring, d_tab,
                                     use_split ? part : nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            auto& dst = use_split ? split : direct;
            ASSERT_EQ(cudaMemcpy(dst.data(), dout.data, dst.size() * sizeof(__half), cudaMemcpyDeviceToHost),
                      cudaSuccess);
        }
        double diff2 = 0.0, ref2 = 0.0;
        for (size_t i = 0; i < q.size(); ++i) {
            const double a = __half2float(direct[i]), b = __half2float(split[i]);
            diff2 += (a - b) * (a - b);
            ref2 += a * a;
        }
        EXPECT_GT(ref2, 0.0);
        EXPECT_LT(std::sqrt(diff2 / ref2), 1e-3) << "rows=" << rows;
        ASSERT_EQ(cudaFree(d_tab), cudaSuccess);
    }
    ASSERT_EQ(cudaFree(part), cudaSuccess);
}

}  // namespace
