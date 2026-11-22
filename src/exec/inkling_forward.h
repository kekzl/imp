#pragma once
// Inkling2-Loop text forward for one sequence: 48 layer applications per token, own K/V and conv state.

#include "model/inkling_weights.h"

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace imp {

class InklingForward {
public:
    InklingForward() = default;
    InklingForward(const InklingForward&) = delete;
    InklingForward& operator=(const InklingForward&) = delete;
    ~InklingForward();

    // max_tokens: rows of one forward call; max_ctx: positions a sequence may reach.
    // va: tracked allocator (tag inkling_state); null = cudaMalloc.
    [[nodiscard]] bool init(const InklingWeights& w, int max_tokens, int max_ctx, VRAMAllocator* va = nullptr);
    // Forgets the sequence: position 0, zero conv state.
    void reset(cudaStream_t s);
    int position() const { return pos_; }

    // Runs T tokens at positions [position(), position() + T). logits: device fp32 [vocab], softcapped,
    // for the last token; may be null.
    [[nodiscard]] bool forward(const int32_t* d_ids, int T, float* d_logits, cudaStream_t s);

    // Engine path: h [T, hidden] fp16 holds the token embeddings and receives the state the final norm
    // reads. pos0 = position of row 0; 0 starts a new sequence, anything else must continue this one.
    [[nodiscard]] bool run(half* h, int T, int pos0, cudaStream_t s);

    // After every layer application: (application index, residual stream [T, hidden]); -1 = embedding,
    // n_apps = the normed final state (only when logits are asked for).
    std::function<void(int, const float*, int)> tap;

private:
    void apply_(int layer, int pass, int T, cudaStream_t s);
    // Embedding norm, every layer application and the final block mix: h_ -> mix_[0].
    void layers_(int T, cudaStream_t s);
    void lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s);
    const float* gate_(const float* h, const float* msg, const InklingGate& g, float* out, int T, cudaStream_t s);
    void conv_(const half* x_h, const InklingConv& cv, const half* xin, int app, int site, bool is_key, int T,
               cudaStream_t s);
    void* alloc_(size_t bytes);
    float* F_(size_t n) { return static_cast<float*>(alloc_(n * sizeof(float))); }
    half* Hf_(size_t n) { return static_cast<half*>(alloc_(n * sizeof(half))); }

    const InklingWeights* w_ = nullptr;
    VRAMAllocator* va_ = nullptr;
    int max_tokens_ = 0, max_ctx_ = 0, ring_ = 0, pos_ = 0, nb_ = 0, app_seq_ = 0;
    bool ok_ = true;
    std::vector<void*> owned_;
    std::vector<half*> k_, v_;          // per application
    std::vector<float*> conv_state_;    // per application: k [3, KD], v [3, KD], attn [3, H], mlp [3, H]
    size_t conv_state_floats_ = 0;
    // fp32 [max_tokens, hidden]
    float *h_ = nullptr, *resid_ = nullptr, *bs_[4] = {}, *pre_h_ = nullptr, *msrc_[4] = {}, *mix_[3] = {};
    float *msg_[4] = {}, *g_[3] = {}, *a_ = nullptr, *res2_ = nullptr, *mm_ = nullptr, *xmf_ = nullptr;
    float *mo_ = nullptr, *co_ = nullptr, *of_ = nullptr;
    int n_msrc_ = 0;
    float *qf_ = nullptr, *rf_ = nullptr, *kf_ = nullptr, *kc_ = nullptr, *attn_ = nullptr, *wts_ = nullptr;
    half *xq_ = nullptr, *xk_ = nullptr, *xv_ = nullptr, *q_h_ = nullptr, *r_h_ = nullptr, *sink_h_ = nullptr;
    half *k_h_ = nullptr, *dyn_ = nullptr, *gate_h_ = nullptr, *ao_h_ = nullptr, *o_h_ = nullptr;
    half *cat_ = nullptr, *gd_ = nullptr, *gf_ = nullptr, *gg_ = nullptr, *xm_ = nullptr;
    half *gu_ = nullptr, *act_ = nullptr, *ey_ = nullptr, *logits_h_ = nullptr;
    int32_t* idx_ = nullptr;
};

}  // namespace imp
