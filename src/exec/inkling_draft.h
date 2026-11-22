#pragma once
// Inkling2 DSpark draft (mtp.safetensors): the model behind InklingForward's decode lookahead. An
// anchor token plus block - 1 mask rows run through a small dense transformer that attends context
// K/V made from the target's feature rows, then a recurrent Markov head picks the tokens one after
// the other. Reference: vLLM vllm/models/inkling2/nvidia/dspark.py and speculator.py.

#include "exec/inkling_forward.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace imp {

class VRAMAllocator;

struct InklingDraftConfig {
    int hidden = 2304, layers = 3, heads = 24, kv_heads = 8, head_dim = 128, inter = 9216;
    int block = 8, rank = 256, window = 1024, vocab_padded = 201024;
    float eps = 1e-6f, theta = 1e6f;
    std::vector<int> taps;    // target visits whose input stream is a feature; the last = final state
    std::vector<bool> local;  // per layer: attends the last window positions before the block only
};

// The target's LM head as per-row E4M3 with fp32 row scales (GraphExecutor::lm_head_fp8_view).
struct InklingHeadView {
    const void* w8 = nullptr;
    const float* row_scales = nullptr;
    int vocab = 0;
};

// Weights, shared by every sequence slot. Allocates under the tag inkling_draft.
struct InklingDraftWeights {
    struct Layer {
        float *in_norm = nullptr, *post_norm = nullptr, *dflare = nullptr;
        float *q_norm = nullptr, *k_norm = nullptr, *k_norm_ctx = nullptr;
        half *qkv = nullptr, *kv_ctx = nullptr, *wo = nullptr, *gate_up = nullptr, *down = nullptr;
    };
    InklingDraftConfig cfg;
    std::vector<Layer> layers;
    float *norm = nullptr, *markov_dflare = nullptr, *alpha = nullptr;  // alpha: [layers + 1][taps]
    float *rnn_b = nullptr, *fold_b = nullptr;
    half *mask = nullptr, *rnn_state_w = nullptr, *rnn_ctx_w = nullptr, *w1 = nullptr, *fold_w = nullptr;
    const float* embed_norm = nullptr;  // the target's, not owned
    InklingHeadView head;
    std::vector<void*> owned;
    VRAMAllocator* va = nullptr;

    InklingDraftWeights() = default;
    InklingDraftWeights(const InklingDraftWeights&) = delete;
    InklingDraftWeights& operator=(const InklingDraftWeights&) = delete;
    ~InklingDraftWeights();
    // dir: the checkpoint directory (config.json mtp_config + mtp.safetensors). False = no draft.
    [[nodiscard]] bool load(const std::string& dir, VRAMAllocator& va, const float* embed_norm,
                            const InklingHeadView& head, cudaStream_t s);
};

// One sequence's draft context. Allocates under the tag inkling_draft_ctx.
class InklingDraft {
public:
    InklingDraft() = default;
    InklingDraft(const InklingDraft&) = delete;
    InklingDraft& operator=(const InklingDraft&) = delete;
    ~InklingDraft();

    // max_ctx positions; cap = most rows one on_rows call brings.
    [[nodiscard]] bool init(const InklingDraftWeights& w, VRAMAllocator& va, int max_ctx, int cap);
    // The two halves of InklingSpecHooks, see there.
    void on_rows(const float* feat, int cap, int n, int pos0, cudaStream_t s);
    int propose(int32_t anchor, const half* d_emb, int pos, int32_t* out, cudaStream_t s);
    InklingSpecHooks hooks();

private:
    void* alloc_(size_t bytes);
    void lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s);
    void convert_(const float* feat, size_t plane, int row0, int n, int pos0, cudaStream_t s);
    void flush_(cudaStream_t s);

    const InklingDraftWeights* w_ = nullptr;
    VRAMAllocator* va_ = nullptr;
    std::vector<void*> owned_;
    bool ok_ = true;
    int max_ctx_ = 0, cap_ = 0;
    // Feature rows kept in pend_ ([taps][cap][hidden]): positions [kept_pos0_, kept_pos0_ + kept_), the
    // first done_ of them already have context K/V. The Markov head reads the row before the anchor.
    int kept_pos0_ = 0, kept_ = 0, done_ = 0;
    std::vector<half*> k_, v_;        // per layer [max_ctx + block][kv_heads * head_dim]
    float *pend_ = nullptr, *h_ = nullptr, *state_ = nullptr, *work_ = nullptr;
    half *xc_ = nullptr, *kvc_ = nullptr, *xn_ = nullptr, *qkv_ = nullptr, *q_ = nullptr, *o_ = nullptr;
    half *y_ = nullptr, *gu_ = nullptr, *act_ = nullptr, *hid_ = nullptr, *logits_ = nullptr, *x2_ = nullptr;
    half *rc_ = nullptr, *latent_ = nullptr, *bias_ = nullptr, *frnn_ = nullptr;
    int32_t* d_tok_ = nullptr;  // [0] the anchor, [1..] the drafted tokens
    float* d_conf_ = nullptr;   // softmax probability of every drafted token
    float *raw_ = nullptr, *latent_f_ = nullptr, *val_ = nullptr;  // recurrent step scratch
    float *cand0_ = nullptr, *mass_ = nullptr;  // candidates' unscored logits, vocabulary mass per span
    int32_t* d_cand_ = nullptr;                 // per draft row: candidate token ids, -1 = none
};

}  // namespace imp
