#pragma once
// Inkling2-Loop text forward for one sequence: 48 layer applications per token, own K/V and conv state.

#include "compute/inkling_ops.h"
#include "core/cuda_raii.h"
#include "model/inkling_weights.h"

#include <cuda_runtime.h>

#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace imp {

// Draft model of the decode lookahead (exec/inkling_draft.h in the engine, a stub in tests).
struct InklingSpecHooks {
    // Feature rows of the positions [pos0, pos0 + n): fp32 planes [n_features][cap][hidden], the
    // residual stream at the tapped applications and the normed final state. Rows of a lookahead
    // call arrive before they are confirmed; a later call at the same position replaces them.
    std::function<void(const float* feat, int cap, int n, int pos0, cudaStream_t s)> on_rows;
    // Up to kGraphRows - 1 tokens expected after anchor (the token at pos, d_emb its embedding row,
    // device fp16 [hidden]); returns their count, 0 = no lookahead for this step.
    std::function<int(int32_t anchor, const half* d_emb, int pos, int32_t* out, cudaStream_t s)> propose;
};

class InklingForward {
public:
    InklingForward() = default;
    InklingForward(const InklingForward&) = delete;
    InklingForward& operator=(const InklingForward&) = delete;
    ~InklingForward();

    // max_tokens: rows of one forward call; max_ctx: positions a sequence may reach.
    // Allocates from va under the tag inkling_state.
    // scratch: a forward of the same sizes whose per-call buffers this one uses too (the slots of a
    // runtime run one after the other on one stream); it must outlive this one.
    [[nodiscard]] bool init(const InklingWeights& w, VRAMAllocator& va, int max_tokens, int max_ctx,
                            const InklingForward* scratch = nullptr);
    // Forgets the sequence: position 0, zero conv state.
    void reset(cudaStream_t s);
    int position() const { return pos_; }

    // Prefix reuse: moves the sequence to the last conv-state checkpoint (kGrid positions apart) that
    // tokens shares with it and returns that position, 0 = from scratch. The next run() starts there.
    [[nodiscard]] int rewind(std::span<const int32_t> tokens, cudaStream_t s);
    // True when a checkpoint exists: recycling the slot would lose a prefix another prompt could reuse.
    [[nodiscard]] bool reusable() const { return ckpt_valid_ > 0; }
    [[nodiscard]] bool has_lookahead() const { return static_cast<bool>(spec_.propose); }
    // The position rewind(tokens) would return; changes nothing.
    [[nodiscard]] int match(std::span<const int32_t> tokens) const;
    static constexpr int kGraphRows = 8;  // rows of the largest captured call = anchor + drafts
    static constexpr int kGrid = 256;     // checkpoint distance
    // Rows per forward segment: a 3111-token prefill ran 2590 tok/s at 256 and 3688 at 1024.
    static constexpr int kSeg = 1024;
    static constexpr int kSlack = 5120;  // sliding K/V kept behind the window: how far rewind may go back

    // Runs T tokens at positions [position(), position() + T). logits: device fp32 [vocab], softcapped,
    // for the last token; may be null.
    [[nodiscard]] bool forward(const int32_t* d_ids, int T, float* d_logits, cudaStream_t s);

    // Engine path: h [T, hidden] fp16 holds the token embeddings and receives the state the final norm
    // reads. pos0 = position of row 0; 0 starts a new sequence, anything else must continue this one.
    // ids (host, may be null): the rows' tokens, recorded for rewind() past the prompt it was given.
    // image_token >= 0: rows with that id hold spliced image features, which skip the embedding norm.
    [[nodiscard]] bool run(half* h, int T, int pos0, cudaStream_t s, const int32_t* ids = nullptr,
                           int32_t image_token = -1);

    // Decode lookahead: a one-row call drafts the next tokens, runs anchor + drafts as one call and
    // serves the following one-row calls from it while their tokens equal the drafts; the first other
    // token puts the conv state back to the last confirmed row. taps: visit indices whose input
    // stream is a feature plane (the last entry stands for the normed final state); embed: the token
    // embedding table [vocab, hidden] the draft rows are read from.
    [[nodiscard]] bool enable_lookahead(InklingSpecHooks hooks, std::vector<int> taps, const half* embed);
    // Lookahead steps, draft rows run, draft rows confirmed since init.
    struct SpecStats {
        long steps = 0, drafted = 0, accepted = 0;
        long confirmed[kGraphRows] = {};  // steps that ended with this many draft rows confirmed
        long paused = 0;                  // one-row steps taken while the draft was paused
    };
    SpecStats spec_stats() const { return stats_; }
    // Tests: off = one-row calls launch every conv site on its own, as calls of more rows do.
    void set_fused_sites(bool on) { fuse_ = on; }
    // Tests: off = many-row calls run the experts slot by slot.
    void set_grouped_experts(bool on) { grouped_ = on; }
    // Tests: off = many-row calls run the row attention kernel.
    void set_gemm_attention(bool on) { attn_gemm_ = on; }

    // After every layer application: (application index, residual stream [T, hidden]); -1 = embedding,
    // n_apps = the normed final state (only when logits are asked for).
    std::function<void(int, const float*, int)> tap;

private:
    void apply_(int layer, int pass, int T, cudaStream_t s);
    void mix_looped_(int layer, int T, cudaStream_t s);
    // The MoE conv site of the last application, not launched yet: the next mix over h_ takes it
    // (take_tail_), anything else that reads h_ launches it first (flush_tail_).
    inkling::MixPre take_tail_();
    void flush_tail_(cudaStream_t s);
    // Embedding norm, every layer application and the final block mix: h_ -> mix_[0].
    void layers_(int T, cudaStream_t s, const int32_t* ids = nullptr, int32_t image_token = -1);
    // Expert projections of a many-row call over expert-sorted slots (NVFP4 layers): xm_ -> ey_.
    // false = not served, nothing written.
    [[nodiscard]] bool moe_rows_(const InklingLayer& ly, int T, cudaStream_t s);
    void experts_(const InklingLayer& ly, int T, cudaStream_t s);
    // 2..8 rows, MXFP8 layers: one tensor-core tile per chosen expert. false = not served.
    [[nodiscard]] bool moe_small_(const InklingLayer& ly, int T, cudaStream_t s);
    void lin_(const half* x, int T, int K, const half* w, int N, half* y, cudaStream_t s);
    void* alloc_(size_t bytes);
    cudaStream_t fork_(cudaStream_t s, int i);
    void join_(cudaStream_t side, cudaStream_t s, int i);
    // Host copy of the conv state when the sequence stands on the grid.
    void checkpoint_(cudaStream_t s, const float* state);
    void checkpoint_at_(int grid, cudaStream_t s, const float* state);
    // Grid rows inside the running many-row segment, for the conv site at off floats into app's state.
    [[nodiscard]] inkling::ConvMarks marks_(size_t app, size_t off) const;
    // run() for one row with the lookahead on; settle_ makes the conv state the confirmed one.
    [[nodiscard]] bool step_spec_(half* h, int32_t tok, cudaStream_t s);
    void settle_(cudaStream_t s);
    void conv_at_(int row, float* dst, cudaStream_t s);
    [[nodiscard]] bool draft_due_();
    // A request starts from the same draft policy state whatever the slot served before: the same prompt
    // then drafts at the same steps (a resend at temperature 0 gave two texts otherwise).
    void reset_draft_policy_() { mean_hits_ = mean_rows_ = 0.0f, tried_ = pause_ = 0, drafted_ = false; }
    [[nodiscard]] bool segment_(half* hs, int seg, cudaStream_t s, const int32_t* ids, int32_t image_token);
    struct ConvLog {
        float *k = nullptr, *v = nullptr, *attn = nullptr, *mlp = nullptr;
    };
    ConvLog conv_log_at_(size_t app, int T) const;
    void tap_features_(int T, cudaStream_t s);
    void capture_decode_(int T, cudaStream_t s);
    float* F_(size_t n) { return static_cast<float*>(alloc_(n * sizeof(float))); }
    half* Hf_(size_t n) { return static_cast<half*>(alloc_(n * sizeof(half))); }

    const InklingWeights* w_ = nullptr;
    VRAMAllocator* va_ = nullptr;
    int max_tokens_ = 0, max_ctx_ = 0, seg_ = 0, ring_ = 0, pos_ = 0, nb_ = 0, app_seq_ = 0;
    bool ok_ = true;
    std::vector<void*> owned_;
    // Per-call buffers in allocation order: a forward given as scratch hands them out again.
    std::vector<void*> scratch_;
    const InklingForward* share_ = nullptr;
    size_t scratch_i_ = 0;
    bool scratch_on_ = false, attn_gemm_ = true;
    float* ckpt_stage_ = nullptr;  // conv states at up to 3 grid rows inside a segment
    int marks_n_ = 0, marks_row_[3] = {};
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
    void* attn_work_ = nullptr;
    size_t attn_work_bytes_ = 0;
    float* part_ = nullptr;                   // decode attention partials (attention_part_floats)
    float *mixp_ = nullptr, *mixp2_ = nullptr;  // block-mix partial sums (mix_partial_floats), one per stream
    // Two side streams for chains that do not depend on each other (fork_ / join_).
    CudaStream side_[2];
    CudaEvent fork_ev_[2], join_ev_[2];
    float *wts_ = nullptr, *rlog_ = nullptr;  // routing weights [rows * top_k], router logits [rows, experts]
    half *xq_ = nullptr, *xk_ = nullptr, *xv_ = nullptr, *o_h_ = nullptr, *ao_h_ = nullptr;
    half *proj_h_ = nullptr, *pk_ = nullptr, *pv_ = nullptr;  // stacked projection outputs
    half* xm_ = nullptr;
    half *gu_ = nullptr, *act_ = nullptr, *ey_ = nullptr, *logits_h_ = nullptr;
    int32_t* idx_ = nullptr;
    half* xs_ = nullptr;  // expert-sorted slot rows [slots, hidden]
    inkling::GateWork gate_work_;
    int32_t *perm_ = nullptr, *off_ = nullptr, *grp_ = nullptr;
    bool grouped_ = true;
    int* d_pos_ = nullptr;  // device: position of row 0 of the running call
    // Captured steps of 1..kGraphRows rows over the own row buffer dec_h_ (run()): any caller rows
    // replay them. One row is the decode step; more are verify chunks and paced prefill.
    CudaGraph decode_graph_[kGraphRows];
    CudaGraphExec decode_exec_[kGraphRows];
    half* dec_h_ = nullptr;
    bool decode_graph_ok_ = true;
    bool fuse_ = true;
    inkling::MixPre tail_;
    // Lookahead (enable_lookahead): feature planes of the running call, the conv state before the
    // stored call and that call's conv inputs, its final rows and tokens.
    InklingSpecHooks spec_;
    std::vector<int> taps_;
    const half* embed_ = nullptr;
    float *feat_ = nullptr, *conv_pre_ = nullptr, *conv_log_ = nullptr, *conv_tmp_ = nullptr;
    half* look_h_ = nullptr;
    int32_t* d_look_ids_ = nullptr;
    int look_pos0_ = -1, look_n_ = 0, look_hits_ = 0;
    // Pay-off of the draft: running means of confirmed and drafted rows per drafting step, steps since
    // the draft was (re)started, and the plain steps left while it is paused.
    float mean_hits_ = 0.0f, mean_rows_ = 0.0f;
    int tried_ = 0, pause_ = 0;
    bool drafted_ = false;  // the last step asked the draft
    int32_t look_tok_[kGraphRows] = {};
    SpecStats stats_;
};

// Weights and the forwards that point into them (destroyed first): one kept sequence per slot.
class InklingDraft;
struct InklingDraftWeights;
struct InklingRuntime {
    InklingWeights w;
    // Decode lookahead (exec/inkling_draft.h): shared draft weights, one context per slot; a slot
    // whose context did not fit decodes without. Declared before the slots that hold hooks into them.
    std::shared_ptr<InklingDraftWeights> draft_w;
    std::vector<std::shared_ptr<InklingDraft>> drafts;
    bool draft_tried = false;
    int max_ctx = 0, cap = 0;
    std::vector<std::unique_ptr<InklingForward>> slots;
    std::vector<int> slot_req;        // request bound to the slot, -1 = never bound
    std::vector<int> slot_block;      // first KV block of that request's table: the key of a decode row
    std::vector<uint64_t> slot_used;  // bind order, the oldest free slot is recycled first
    uint64_t tick = 0;
    std::vector<int32_t> ids;  // host copies of one call's token ids, positions, first blocks
    std::vector<int> pos, blk;
    // The three row arrays of a call packed on the device ([3][rows_cap]) and their host copy.
    int32_t* d_rows = nullptr;
    int rows_cap = 0;
    std::vector<int32_t> rows;
    VRAMAllocator* va = nullptr;
    InklingRuntime() = default;
    InklingRuntime(const InklingRuntime&) = delete;
    InklingRuntime& operator=(const InklingRuntime&) = delete;
    ~InklingRuntime();
    // Slot of a request / of a decode row's first block, -1 when unknown.
    [[nodiscard]] int slot_of_req(int req) const;
    [[nodiscard]] int slot_of_block(int block) const;
    // Binds req to the free slot (busy[i] false) that shares the longest prefix with tokens; on a tie a
    // slot without a checkpoint goes first, then one with the decode lookahead, then the oldest binding.
    // Returns the slot, -1 = all busy.
    [[nodiscard]] int bind(int req, std::span<const int32_t> tokens, const std::vector<bool>& busy);
};

class Model;
// Null for every other arch; throws when an Inkling2-Loop checkpoint cannot be loaded.
// Up to slots (<= 0: two) sequences of max_ctx positions, max_tokens rows per call; fewer when the
// card does not take them all, at least one.
std::unique_ptr<InklingRuntime> make_inkling_runtime(const Model& model, VRAMAllocator* va, int max_tokens,
                                                     int max_ctx, int slots);

}  // namespace imp
