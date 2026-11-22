#pragma once
// Inkling2-Loop text forward for one sequence: 48 layer applications per token, own K/V and conv state.

#include "core/cuda_raii.h"
#include "model/inkling_weights.h"

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace imp {

class InklingForward {
public:
    InklingForward() = default;
    InklingForward(const InklingForward&) = delete;
    InklingForward& operator=(const InklingForward&) = delete;
    ~InklingForward();

    // max_tokens: rows of one forward call; max_ctx: positions a sequence may reach.
    // Allocates from va under the tag inkling_state.
    [[nodiscard]] bool init(const InklingWeights& w, VRAMAllocator& va, int max_tokens, int max_ctx);
    // Forgets the sequence: position 0, zero conv state.
    void reset(cudaStream_t s);
    int position() const { return pos_; }

    // Prefix reuse: moves the sequence to the last conv-state checkpoint (kGrid positions apart) that
    // tokens shares with it and returns that position, 0 = from scratch. The next run() starts there.
    [[nodiscard]] int rewind(std::span<const int32_t> tokens, cudaStream_t s);
    // True when a checkpoint exists: recycling the slot would lose a prefix another prompt could reuse.
    [[nodiscard]] bool reusable() const { return ckpt_valid_ > 0; }
    // The position rewind(tokens) would return; changes nothing.
    [[nodiscard]] int match(std::span<const int32_t> tokens) const;
    static constexpr int kGrid = 256;    // rows per forward segment = checkpoint distance
    static constexpr int kSlack = 5888;  // sliding K/V kept behind the window: how far rewind may go back

    // Runs T tokens at positions [position(), position() + T). logits: device fp32 [vocab], softcapped,
    // for the last token; may be null.
    [[nodiscard]] bool forward(const int32_t* d_ids, int T, float* d_logits, cudaStream_t s);

    // Engine path: h [T, hidden] fp16 holds the token embeddings and receives the state the final norm
    // reads. pos0 = position of row 0; 0 starts a new sequence, anything else must continue this one.
    // ids (host, may be null): the rows' tokens, recorded for rewind() past the prompt it was given.
    // image_token >= 0: rows with that id hold spliced image features, which skip the embedding norm.
    [[nodiscard]] bool run(half* h, int T, int pos0, cudaStream_t s, const int32_t* ids = nullptr,
                           int32_t image_token = -1);

    // After every layer application: (application index, residual stream [T, hidden]); -1 = embedding,
    // n_apps = the normed final state (only when logits are asked for).
    std::function<void(int, const float*, int)> tap;

private:
    void apply_(int layer, int pass, int T, cudaStream_t s);
    // Embedding norm, every layer application and the final block mix: h_ -> mix_[0].
    void layers_(int T, cudaStream_t s, const int32_t* ids = nullptr, int32_t image_token = -1);
    void lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s);
    void* alloc_(size_t bytes);
    // Host copy of the conv state when the sequence stands on the grid.
    void checkpoint_(cudaStream_t s);
    void capture_decode_(cudaStream_t s);
    float* F_(size_t n) { return static_cast<float*>(alloc_(n * sizeof(float))); }
    half* Hf_(size_t n) { return static_cast<half*>(alloc_(n * sizeof(half))); }

    const InklingWeights* w_ = nullptr;
    VRAMAllocator* va_ = nullptr;
    int max_tokens_ = 0, max_ctx_ = 0, seg_ = 0, ring_ = 0, pos_ = 0, nb_ = 0, app_seq_ = 0;
    bool ok_ = true;
    std::vector<void*> owned_;
    std::vector<half*> k_, v_;        // per application
    std::vector<float*> conv_state_;  // per application: k [3, KD], v [3, KD], attn [3, H], mlp [3, H]
    size_t conv_state_floats_ = 0;
    float* conv_all_ = nullptr;             // the n_apps conv states, contiguous
    std::vector<std::vector<float>> ckpt_;  // [g - 1] = conv state at position g * kGrid
    int ckpt_valid_ = 0;                    // checkpoints 1..ckpt_valid_ belong to this sequence
    int hw_ = 0;                            // highest position written since the last start at 0
    std::vector<int32_t> hist_;             // tokens of the sequence; rows [0, pos_) are computed
    bool armed_ = false;                    // rewind() set hist_ for the run that follows
    // fp32 [max_tokens, hidden]
    float *h_ = nullptr, *bs_[4] = {}, *pre_h_ = nullptr, *msrc_[4] = {}, *mix_[3] = {};
    float *msg_[4] = {}, *g_[3] = {}, *gd_ = nullptr, *res2_ = nullptr,
          *mm_ = nullptr;  // g_, gd_: gate scratch
    int n_msrc_ = 0;
    float* part_ = nullptr;                   // decode attention partials (attention_part_floats)
    float* mixp_ = nullptr;                   // block-mix partial sums (mix_partial_floats)
    float *wts_ = nullptr, *rlog_ = nullptr;  // routing weights [rows * top_k], router logits [rows, experts]
    half *xq_ = nullptr, *xk_ = nullptr, *xv_ = nullptr, *o_h_ = nullptr, *ao_h_ = nullptr;
    half *proj_h_ = nullptr, *pk_ = nullptr, *pv_ = nullptr;  // stacked projection outputs
    half* xm_ = nullptr;
    half *gu_ = nullptr, *act_ = nullptr, *ey_ = nullptr, *logits_h_ = nullptr;
    int32_t* idx_ = nullptr;
    int* d_pos_ = nullptr;  // device: position of row 0 of the running call
    // Captured one-token step over the own row buffer dec_h_ (run(), T == 1): any caller row replays it.
    CudaGraph decode_graph_;
    CudaGraphExec decode_exec_;
    half* dec_h_ = nullptr;
    bool decode_graph_ok_ = true;
};

// Weights and the forwards that point into them (destroyed first): one kept sequence per slot.
struct InklingRuntime {
    InklingWeights w;
    std::vector<std::unique_ptr<InklingForward>> slots;
    std::vector<int> slot_req;        // request bound to the slot, -1 = never bound
    std::vector<int> slot_block;      // first KV block of that request's table: the key of a decode row
    std::vector<uint64_t> slot_used;  // bind order, the oldest free slot is recycled first
    uint64_t tick = 0;
    std::vector<int32_t> ids;  // host copies of one call's token ids, positions, first blocks
    std::vector<int> pos, blk;
    // Slot of a request / of a decode row's first block, -1 when unknown.
    [[nodiscard]] int slot_of_req(int req) const;
    [[nodiscard]] int slot_of_block(int block) const;
    // Binds req to the free slot (busy[i] false) that shares the longest prefix with tokens; on a tie a
    // slot without a checkpoint goes first, then the oldest binding. Returns the slot, -1 = all busy.
    [[nodiscard]] int bind(int req, std::span<const int32_t> tokens, const std::vector<bool>& busy);
};

class Model;
// Null for every other arch; throws when an Inkling2-Loop checkpoint cannot be loaded.
// Up to slots (<= 0: two) sequences of max_ctx positions, max_tokens rows per call; fewer when the
// card does not take them all, at least one.
std::unique_ptr<InklingRuntime> make_inkling_runtime(const Model& model, VRAMAllocator* va, int max_tokens,
                                                     int max_ctx, int slots);

}  // namespace imp
