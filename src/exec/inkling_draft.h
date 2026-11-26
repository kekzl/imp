#pragma once
// Inkling2 DSpark draft (mtp.safetensors): the model behind InklingForward's decode lookahead. An
// anchor token plus block - 1 mask rows run through a small dense transformer that attends context
// K/V made from the target's feature rows, then a recurrent Markov head picks the tokens one after
// the other. Reference: vLLM vllm/models/inkling2/nvidia/dspark.py and speculator.py.

#include "core/cuda_raii.h"
#include "exec/inkling_forward.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

class VRAMAllocator;

// dir/config.json's mtp_config (InklingDraftConfig in model/inkling_config.h); false = none or unusable.
[[nodiscard]] bool parse_inkling_draft_config(const std::string& dir, InklingDraftConfig& c);

// The target's LM head as per-row E4M3 with fp32 row scales (GraphExecutor::lm_head_fp8_view).
struct InklingHeadView {
    const void* w8 = nullptr;
    const float* row_scales = nullptr;
    int vocab = 0;
};

// Weights, shared by every sequence slot. Allocates under the tag inkling_draft.
struct InklingDraftWeights {
    // A block projection as I8B32 (compute/gemv_i8b32.h): the draft reads 495 MB of fp16 projections
    // per call otherwise, and its tokens are only proposals.
    struct Q8 {
        int8_t* q = nullptr;
        half* scales = nullptr;
    };
    struct Layer {
        float *in_norm = nullptr, *post_norm = nullptr, *dflare = nullptr;
        float *q_norm = nullptr, *k_norm = nullptr, *k_norm_ctx = nullptr;
        // qkv, wo, gate_up, down: null once their I8B32 form exists.
        half *qkv = nullptr, *kv_ctx = nullptr, *wo = nullptr, *gate_up = nullptr, *down = nullptr;
        Q8 qkv8, wo8, gate_up8, down8;
    };
    InklingDraftConfig cfg;
    std::vector<Layer> layers;
    float *norm = nullptr, *markov_dflare = nullptr, *alpha = nullptr;  // alpha: [layers + 1][taps]
    float *rnn_b = nullptr, *fold_b = nullptr;
    half *mask = nullptr, *rnn_state_w = nullptr, *rnn_ctx_w = nullptr, *w1 = nullptr, *fold_w = nullptr;
    const float* embed_norm = nullptr;  // the target's, not owned
    InklingHeadView head;
    std::vector<VramOwned<std::byte>> owned;
    VRAMAllocator* va = nullptr;

    InklingDraftWeights() = default;
    InklingDraftWeights(const InklingDraftWeights&) = delete;
    InklingDraftWeights& operator=(const InklingDraftWeights&) = delete;
    ~InklingDraftWeights();
    // dir: the checkpoint directory (config.json mtp_config + mtp.safetensors). False = no draft.
    [[nodiscard]] bool load(const std::string& dir, VRAMAllocator& va, const float* embed_norm,
                            const InklingHeadView& head, cudaStream_t s);

private:
    // Also the load's last synchronize: false = a device error.
    [[nodiscard]] bool quantize_blocks_(cudaStream_t s);
};

// One sequence's draft context. Allocates under the tag inkling_draft_ctx.
class InklingDraft {
public:
    InklingDraft() = default;
    InklingDraft(const InklingDraft&) = delete;
    InklingDraft& operator=(const InklingDraft&) = delete;
    ~InklingDraft();

    // max_ctx positions; cap = most rows one on_rows call brings; back = how far before the highest
    // position seen a later on_rows may start again (InklingForward keeps kSeg + kSlack), < 0 = every
    // layer keeps one K/V row per position.
    // own_kv: the full-context layer keeps max_ctx + block rows here; false = a pool layer (bind_kv), the
    // local layers then always keep a ring.
    // look: a lookahead region of the engine's slab; the local rings live there (its draft_ring must be
    // this ring), empty = allocated here.
    [[nodiscard]] bool init(const InklingDraftWeights& w, VRAMAllocator& va, int max_ctx, int cap,
                            int back = InklingForward::kSeg + InklingForward::kSlack, bool own_kv = true,
                            const InklingLookRegion& look = {});
    // The full-context layer in the engine's pool: k / v its bases, table the slot's block table
    // (InklingForward::table()). False: not exactly one such layer, the draft stays off.
    [[nodiscard]] bool bind_kv(half* k, half* v, const int* table);
    // The parts of InklingSpecHooks, see there.
    void on_rows(const float* feat, int cap, int n, int pos0, cudaStream_t s);
    int propose(int32_t anchor, const half* d_emb, int pos, int32_t* out, cudaStream_t s);
    void restart(int keep);
    InklingSpecHooks hooks();

private:
    void* alloc_(size_t bytes);
    void lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s);
    // The block projection w8, or the fp16 one when the I8B32 form does not exist.
    void lin8_(const half* x, int T, int K, const InklingDraftWeights::Q8& w8, const half* w, int N, half* y,
               cudaStream_t s);
    void convert_(const float* feat, size_t plane, int row0, int n, int pos0, cudaStream_t s);
    void flush_(cudaStream_t s);
    [[nodiscard]] bool block_(cudaStream_t s);
    void capture_(cudaStream_t s);

    const InklingDraftWeights* w_ = nullptr;
    VRAMAllocator* va_ = nullptr;
    std::vector<VramOwned<std::byte>> owned_;
    // The full-context layer's block table (pool K/V, null = own rows) and its index, -1 = none.
    const int* table_ = nullptr;
    int global_ = -1;
    bool bound_ = false;  // every layer's K/V have memory
    bool ok_ = true;
    int max_ctx_ = 0, cap_ = 0;
    int ring_ = 0;  // rows of a local layer's K/V ring (kept twice), 0 = one row per position
    // Feature rows kept in pend_ ([taps][cap][hidden]): positions [kept_pos0_, kept_pos0_ + kept_), the
    // first done_ of them already have context K/V. The Markov head reads the row before the anchor.
    int kept_pos0_ = 0, kept_ = 0, done_ = 0;
    int drafted_pos_ = -1;  // position of the last propose(): its call's rows get their K/V in on_rows
    std::vector<half*> k_, v_;  // per layer [max_ctx + block or 2 * ring_][kv_heads * head_dim]
    float *pend_ = nullptr, *h_ = nullptr, *state_ = nullptr, *work_ = nullptr;
    half *xc_ = nullptr, *kvc_ = nullptr, *xn_ = nullptr, *qkv_ = nullptr, *q_ = nullptr, *o_ = nullptr;
    half *y_ = nullptr, *gu_ = nullptr, *act_ = nullptr, *hid_ = nullptr, *logits_ = nullptr, *x2_ = nullptr;
    half *rc_ = nullptr, *latent_ = nullptr, *bias_ = nullptr, *frnn_ = nullptr;
    int32_t* d_tok_ = nullptr;  // [0] the anchor, [1..] the drafted tokens
    // What the captured block reads per call: [0] position, [1] local context length, [2] its first
    // key, [3] the feature row before the anchor, [4] the block's row in a local layer's K/V; the
    // anchor's embedding row.
    int* d_par_ = nullptr;
    int par_[5] = {};
    int32_t tok_ = 0;
    half* emb_ = nullptr;
    CudaGraph graph_;
    CudaGraphExec exec_;
    bool graph_ok_ = true;
    uint64_t exec_epoch_ = 0;   // graph_epoch() before the capture (I3)
    float* d_conf_ = nullptr;   // softmax probability of every drafted token
    float *raw_ = nullptr, *latent_f_ = nullptr, *val_ = nullptr;  // recurrent step scratch
    float *cand0_ = nullptr, *mass_ = nullptr;  // candidates' unscored logits, vocabulary mass per span
    int32_t* d_cand_ = nullptr;                 // per draft row: candidate token ids, -1 = none
};

}  // namespace imp
