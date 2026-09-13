// mtp_feed_batch() must match n feed-only mtp_draft_step() calls: identical mtp_pos, matching
// KV cache rows and h_final. Synthetic dense head (Qwen3.8/3.6-27B layout) keeps it model-free;
// RMS-relative tolerance since M=1 GEMV and M=n GEMM reduce in different orders. GPU required.

#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <vector>

#include "compute/mtp_forward.h"
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

// Guardrails: a MoE-shaped workspace must report no batch capability, and a
// feed beyond the KV capacity must refuse rather than write out of bounds.
TEST_F(MtpFeedBatchTest, RefusesUnsupportedOrOverCapacity) {
    SyntheticHead s;
    s.build(/*seed=*/13);

    imp::MtpDraftWorkspace ws;
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws, kH, kVocab, 0, 0, 0, kDff, kNh, kNkv, kHd,
                                            kSeq));
    configure_ws(ws);
    ws.mtp_pos = kSeq - 4;  // only 4 slots left
    EXPECT_FALSE(imp::mtp_feed_batch(s.tokens.data(), s.d_hidden, 8, s.head, s.tok_emb, ws,
                                     kH, nullptr));
    EXPECT_EQ(ws.mtp_pos, kSeq - 4) << "failed feed must not advance the cache";
    imp::mtp_workspace_free(ws);

    // MoE-shaped alloc (n_experts > 0) gets no batch scratch.
    imp::MtpDraftWorkspace ws_moe;
    ASSERT_TRUE(imp::mtp_workspace_allocate(ws_moe, kH, kVocab, /*n_experts=*/4, /*top_k=*/2,
                                            /*expert_d_ff=*/32, kDff, kNh, kNkv, kHd, kSeq));
    EXPECT_EQ(ws_moe.feed_rows_cap, 0);
    imp::mtp_workspace_free(ws_moe);
}

}  // namespace
