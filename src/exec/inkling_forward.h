#pragma once
// Inkling2-Loop text forward for one sequence: 48 layer applications per token, conv state and sliding K/V
// in its slot of the engine's slab (or a slab of its own, standalone), the full-attention K/V in the
// engine's paged pool (or its own, standalone).

#include "compute/inkling_ops.h"
#include "core/cuda_raii.h"
#include "exec/inkling_kv_layout.h"
#include "exec/inkling_slab_layout.h"
#include "exec/inkling_slot_table.h"
#include "memory/host_pinned.h"
#include "memory/snapshot_piece.h"
#include "model/inkling_weights.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
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
    // The sequence goes on from position keep (0 = a new one): rows from there on are void. May be empty.
    std::function<void(int keep)> restart;
    // Cross-slot snapshot (B-111 S4, InklingSnapLayout draft part at entry offset off): the draft's rows of
    // the window before pos as pieces, and back from an entry as a sequence that goes on at pos. May be
    // empty.
    std::function<bool(int pos, size_t off, std::vector<SnapshotPiece>& out, cudaStream_t s)> save;
    std::function<bool(int pos, const std::byte* entry, size_t off, cudaStream_t s)> load;
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
    // own_kv: the full-attention K/V live in max_ctx rows of this forward behind an identity block table
    // (tests, tools); false = the engine's paged pool, bind_kv() and set_table() before run().
    // state: this slot's region of the engine's slab (inkling_slab_layout(w.cfg, max_tokens)), holding the
    // conv state and the sliding rings; null = a one-slot slab of its own.
    [[nodiscard]] bool init(const InklingWeights& w, VRAMAllocator& va, int max_tokens, int max_ctx,
                            const InklingForward* scratch = nullptr, bool own_kv = true,
                            std::byte* state = nullptr);
    // Pool K/V: k[l] / v[l] = base of pool layer l (inkling_kv_apps order, kKVBlockSize positions per
    // block, [pos][n_kv][hd] within one). The regions must outlive this forward and never move.
    [[nodiscard]] bool bind_kv(std::span<half* const> k, std::span<half* const> v);
    // The sequence's block table for the next run(): n entries of d_table (device) copied on s. It must
    // cover position() + T + kInklingKvAhead of that call.
    void set_table(const int* d_table, int n, cudaStream_t s);
    // The table the full-attention K/V are read through (device, table_cap() entries, never moves).
    [[nodiscard]] const int* table() const { return blocks_; }
    [[nodiscard]] int table_cap() const { return n_blocks_; }
    [[nodiscard]] const InklingConfig& config() const { return w_->cfg; }
    // Forgets the sequence: position 0, zero conv state.
    void reset(cudaStream_t s);
    int position() const { return pos_; }

    // Prefix reuse: moves the sequence to the last conv-state checkpoint (kGrid positions apart) that
    // tokens shares with it, at most limit, and returns that position, 0 = from scratch. The next run()
    // starts there; the K/V of the positions before it must be in the table set for that run.
    [[nodiscard]] int rewind(std::span<const int32_t> tokens, cudaStream_t s, int limit = INT_MAX);
    // True when a checkpoint exists: recycling the slot would lose a prefix another prompt could reuse.
    [[nodiscard]] bool reusable() const { return ckpt_valid_ > 0; }
    // Pinned host bytes planned at init for the conv-state checkpoints (inkling_checkpoint_slots of them),
    // and the checkpoints a rewind can still reach.
    [[nodiscard]] size_t checkpoint_bytes() const { return ckpt_host_.bytes(); }
    [[nodiscard]] int checkpoints_held() const;

    // Cross-slot prefix reuse (B-111 S4) through RecurrentSnapshotStore entries of snapshot_bytes()
    // (inkling_snap_layout): the highest checkpoint position whose state this slot can still give for tokens
    // (its history up to there), 0 = none; that state as pieces of one entry, the lookahead draft's rows too
    // (flushed on s), sources valid until the next run(); and restore, which continues tokens on this slot
    // from an entry saved on any slot at pos. with_draft: the entry holds draft rows; a slot with a draft
    // takes no entry without. set_snapshot_draft: entries hold a draft part (every slot of a runtime alike).
    void set_snapshot_draft(const InklingDraftConfig* d);
    [[nodiscard]] size_t snapshot_bytes() const { return snap_.bytes; }
    [[nodiscard]] bool snapshot_has_draft() const {
        return snap_.draft_bytes > 0 && static_cast<bool>(spec_.save);
    }
    [[nodiscard]] bool snapshot_plans_draft() const { return snap_.draft_bytes > 0; }
    [[nodiscard]] int snapshot_pos(std::span<const int32_t> tokens) const;
    [[nodiscard]] bool snapshot_pieces(int pos, std::vector<SnapshotPiece>& out, cudaStream_t s);
    [[nodiscard]] bool restore(std::span<const int32_t> tokens, const void* entry, int pos, cudaStream_t s,
                               bool with_draft = true);
    [[nodiscard]] bool has_lookahead() const { return static_cast<bool>(spec_.propose); }
    // The position rewind(tokens) would return; changes nothing.
    [[nodiscard]] int match(std::span<const int32_t> tokens) const;
    static constexpr int kGraphRows = kInklingGraphRows;  // rows of the largest call = anchor + drafts
    static constexpr int kGrid = 256;     // checkpoint distance
    // Rows per forward segment: a 3111-token prefill ran 2590 tok/s at 256 and 3688 at 1024.
    static constexpr int kSeg = kInklingSeg;
    static constexpr int kSlack = kInklingSlack;  // sliding K/V behind the window: how far rewind may go

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
    // embedding table [vocab, hidden] the draft rows are read from. look: this slot's lookahead region of
    // the engine's slab (planned for taps.size() taps and this segment), empty = allocated here.
    [[nodiscard]] bool enable_lookahead(InklingSpecHooks hooks, std::vector<int> taps, const half* embed,
                                        const InklingLookRegion& look = {});
    // Lookahead steps, draft rows run, draft rows confirmed since init.
    struct SpecStats {
        long steps = 0, drafted = 0, accepted = 0;
        long confirmed[kGraphRows] = {};  // steps that ended with this many draft rows confirmed
        long paused = 0;                  // one-row steps taken while the draft was paused
        long proposals = 0;               // draft calls
        double propose_ms = 0.0;          // their wall time (each ends in a stream synchronize)
    };
    SpecStats spec_stats() const { return stats_; }
    // The last run(): its row in the stored lookahead call (0 = the call itself ran), -1 = no stored call.
    [[nodiscard]] int look_row() const { return look_row_; }
    // That call's states [rows][hidden] (what the final norm reads) and its row count.
    [[nodiscard]] const half* look_states(int& rows) const {
        rows = look_n_;
        return look_h_;
    }
    // Tests: off = one-row calls launch every conv site on its own, as calls of more rows do.
    void set_fused_sites(bool on) { fuse_ = on; }
    // Calls of up to 8 rows read the dense projections as I8B32 from d (set before the first run();
    // null = fp16). d must outlive this forward.
    void set_dense8(const InklingDense8* d) { dense8_ = d; }
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
    [[nodiscard]] bool own_projections_(const InklingLayer& ly, int T) const;
    void* alloc_(size_t bytes);
    // inkling_forward_attn.cu: the block table and, own_kv, the full-attention K/V behind it; the
    // attention of application app over the q, rel, sink, gate rows in proj_h_ (q_stride halves apart).
    [[nodiscard]] bool init_kv_(bool own);
    void attend_(const InklingLayer& ly, size_t app, bool global, int T, int q_stride, cudaStream_t s);
    cudaStream_t fork_(cudaStream_t s, int i);
    void join_(cudaStream_t side, cudaStream_t s, int i);
    // Host copy of the conv state when the sequence stands on the grid.
    void checkpoint_(cudaStream_t s, const float* state);
    void checkpoint_at_(int grid, cudaStream_t s, const float* state);
    // Checkpoint g's conv state in the planned host ring; held_: g is this sequence's and not overwritten.
    [[nodiscard]] float* ckpt_at_(int g) const;
    [[nodiscard]] bool held_(int g) const;
    // Positions of tokens that agree with the history, up to lim.
    [[nodiscard]] int agree_(std::span<const int32_t> tokens, size_t lim) const;
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
    // I3: drops the captured steps of an older graph epoch; the caller captures again as on first use.
    void drop_stale_graphs_(cudaStream_t s);
    float* F_(size_t n) { return static_cast<float*>(alloc_(n * sizeof(float))); }
    half* Hf_(size_t n) { return static_cast<half*>(alloc_(n * sizeof(half))); }

    const InklingWeights* w_ = nullptr;
    const InklingDense8* dense8_ = nullptr;
    VRAMAllocator* va_ = nullptr;
    int max_tokens_ = 0, max_ctx_ = 0, seg_ = 0, ring_ = 0, pos_ = 0, nb_ = 0, app_seq_ = 0;
    bool ok_ = true;
    std::vector<VramOwned<std::byte>> owned_;
    // Block table of the full-attention K/V, kKVBlockSize rows per entry: the pool's (set_table) or, own
    // K/V, the identity over their rows; table_rows_: the positions its last set_table covers (a call and
    // its lookahead rows stay below). kv_bound_: k_ / v_ of those applications point at memory.
    int* blocks_ = nullptr;
    int n_blocks_ = 0, table_rows_ = 0;
    bool kv_bound_ = false;
    // Per-call buffers in allocation order: a forward given as scratch hands them out again.
    std::vector<void*> scratch_;
    const InklingForward* share_ = nullptr;
    size_t scratch_i_ = 0;
    bool scratch_on_ = false, attn_gemm_ = true;
    float* ckpt_stage_ = nullptr;  // conv states at up to 3 grid rows inside a segment
    int marks_n_ = 0, marks_row_[3] = {};
    std::vector<half*> k_, v_;        // per application, views of the slot's rings (null: full attention)
    std::vector<float*> conv_state_;  // per application: k [3, KD], v [3, KD], attn [3, H], mlp [3, H]
    size_t conv_state_floats_ = 0;
    float* conv_all_ = nullptr;  // the n_apps conv states, contiguous at the slot's offset 0
    InklingSlabLayout layout_;   // of the slot conv_all_ starts
    InklingSnapLayout snap_;     // a snapshot entry (set_snapshot_draft adds the draft part)
    // Conv state at position g * kGrid in slot (g - 1) % ckpt_n_ of a pinned ring planned at init (I2, I5).
    PinnedBuffer ckpt_host_;
    int ckpt_n_ = 0;
    int ckpt_valid_ = 0;  // checkpoints ckpt_lo_..ckpt_valid_ belong to this sequence
    int ckpt_free_ = 0;   // checkpoints 1..ckpt_free_ are dead: too far back to resume, or overwritten
    int ckpt_lo_ = 1;     // a restore at checkpoint g: the ones before it are another sequence's
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
    uint64_t decode_epoch_ = 0;  // graph_epoch() every held decode_exec_ was captured under
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
    int look_pos0_ = -1, look_n_ = 0, look_hits_ = 0, look_row_ = -1;
    // Pay-off of the draft: running means of confirmed and drafted rows per drafting step, steps since
    // the draft was (re)started, and the plain steps left while it is paused.
    float mean_hits_ = 0.0f, mean_rows_ = 0.0f;
    int tried_ = 0, pause_ = 0;
    bool drafted_ = false;  // the last step asked the draft
    int32_t look_tok_[kGraphRows] = {};
    SpecStats stats_;
};

static_assert(kInklingKvAhead == InklingForward::kGraphRows - 1, "a lookahead call writes kGraphRows rows");

// Weights and the forwards that point into them (destroyed first): one kept sequence per slot.
class InklingDraft;
class KVCache;
class InklingSlab;
struct InklingDraftWeights;
struct InklingRuntime {
    InklingWeights w;
    // Decode lookahead (exec/inkling_draft.h): shared draft weights, one context per slot; a slot
    // whose context did not fit decodes without. Declared before the slots that hold hooks into them.
    std::shared_ptr<InklingDraftWeights> draft_w;
    std::vector<std::shared_ptr<InklingDraft>> drafts;
    bool draft_tried = false;
    int max_ctx = 0, cap = 0;
    // The paged pool (InklingRuntime::bind_kv): base of every pool layer, the draft's among them
    // (-1 = none); per slot the request its sequence was last rewound for (later prefill chunks go on).
    std::vector<half*> kv_k, kv_v;
    int draft_layer = -1;
    std::vector<int> resumed;
    // I8B32 dense projections of the decode steps (before the slots that point at them); null when
    // the card has no room.
    std::unique_ptr<InklingDense8> dense8;
    std::vector<std::unique_ptr<InklingForward>> slots;
    // Request id -> slot, the key of every row (InferenceState::h_seq_ids, prefill_req_id).
    InklingSlotTable table;
    std::vector<int32_t> ids;  // host copies of one call's token ids and positions, slot per decode row
    std::vector<int> pos, row_slot;
    // The two row arrays of a call packed on the device ([2][rows_cap]) and their host copy.
    VramOwned<int32_t> d_rows;
    int rows_cap = 0;
    std::vector<int32_t> rows;
    // Lookahead slots: logits of every row of the slot's stored call ([kGraphRows][vocab], one LM-head
    // pass for all of them, in the slot's lookahead region of the slab), and the row the running decode
    // step takes instead of its own head pass.
    std::vector<float*> look_logits;
    std::vector<long> look_serial;  // the stored call (its lookahead step count) the slot's logits belong to
    const float* logit_src = nullptr;
    VRAMAllocator* va = nullptr;
    InklingRuntime() = default;
    InklingRuntime(const InklingRuntime&) = delete;
    InklingRuntime& operator=(const InklingRuntime&) = delete;
    ~InklingRuntime();
    // Binds req to the unbound slot that shares the longest prefix with tokens; on a tie a slot
    // without a checkpoint goes first, then one with the decode lookahead, then the oldest binding.
    // Returns the slot, -1 = every slot is bound.
    [[nodiscard]] int bind(int req, std::span<const int32_t> tokens);
    // want slots of ctx positions over the weights wt (pool K/V, bind_kv later), the I8B32 dense projections
    // when inkling_dense8_fits; false = not one slot. slab: allocated here, holds each slot's conv state and
    // sliding rings; it must outlive the slots. make_inkling_runtime and the serve-path test use it.
    // draft (null = no lookahead): the slab also holds a lookahead region of vocab logits per row for the
    // first kInklingLookaheadSlots slots (draft rings, lookahead planes; inkling_attach_draft_ views them).
    [[nodiscard]] bool init_slots(const InklingWeights& wt, VRAMAllocator& alloc, int max_tokens, int ctx,
                                  int want, InklingSlab& slab, const InklingDraftConfig* draft = nullptr,
                                  int vocab = 0);
    // bind, then what that slot can resume of tokens (InklingForward::match); -1 = every slot is bound.
    [[nodiscard]] int bind_match(int req, std::span<const int32_t> tokens);
    // A prefill chunk of req at offset. first: this chunk bound the slot (unless bound) and rewound it to at
    // most offset, whose K/V the request's blocks hold. Returns where the prefill goes on, -1 = no slot.
    // snap: a snapshot entry saved at offset on any slot (InklingForward::restore), restored when the slot's
    // own checkpoints reach less; null = none. snap_draft: it holds the draft's rows
    // (InklingForward::restore). restored: set when this call restored snap.
    [[nodiscard]] int resume(int req, std::span<const int32_t> tokens, int offset, cudaStream_t s,
                             bool& first, const void* snap = nullptr, bool snap_draft = true,
                             bool* restored = nullptr);
    // Every slot's full-attention K/V (and the draft's) in kv: inkling_kv_apps layers, fp16, 16-row blocks.
    [[nodiscard]] bool bind_kv(KVCache& kv);
};

class Model;
// Null for every other arch; throws when an Inkling2-Loop checkpoint cannot be loaded.
// Up to slots (<= 0: two) sequences of max_ctx positions, max_tokens rows per call; fewer when the
// card does not take them all, at least one. Their full-attention K/V live in the engine's paged pool
// (InklingRuntime::bind_kv before the first forward), their conv state and sliding rings in slab (the
// engine's, Engine::inkling_slab_); null throws. lookahead (speculative.mtp_k != 0) with mtp.safetensors and
// its mtp_config: the slab also holds the draft rings and lookahead planes, before serving.
std::unique_ptr<InklingRuntime> make_inkling_runtime(const Model& model, VRAMAllocator* va, int max_tokens,
                                                     int max_ctx, int slots, InklingSlab* slab,
                                                     bool lookahead);

}  // namespace imp
