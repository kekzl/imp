// InklingForward's prefix reuse (inkling_forward.h): the conv-state checkpoints of its own sequence (rewind)
// and the cross-slot snapshots another slot resumes from, the lookahead draft's rows with them (B-111 S4).
// Host code: copies and bookkeeping.

#include "exec/inkling_draft.h"
#include "exec/inkling_forward.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace imp {

void InklingForward::checkpoint_(cudaStream_t s, const float* state) {
    checkpoint_at_(pos_ / kGrid, s, state);
}

float* InklingForward::ckpt_at_(int g) const {
    if (g < 1 || ckpt_n_ < 1)
        return nullptr;
    return ckpt_host_.as<float>() +
           static_cast<size_t>((g - 1) % ckpt_n_) * conv_state_floats_ * conv_state_.size();
}

bool InklingForward::held_(int g) const {
    // Sliding K/V of [g * kGrid - window, g * kGrid) must not have been overwritten by positions up to hw_.
    return g >= 1 && g >= ckpt_lo_ && g <= ckpt_valid_ && g > ckpt_free_ &&
           hw_ - g * kGrid <= ring_ - w_->cfg.window && ckpt_at_(g) != nullptr;
}

int InklingForward::checkpoints_held() const {
    return std::max(ckpt_valid_ - std::max(ckpt_free_, ckpt_lo_ - 1), 0);
}

void InklingForward::checkpoint_at_(int grid, cudaStream_t s, const float* state) {
    const size_t n = conv_state_floats_ * conv_state_.size();
    float* dst = ckpt_at_(grid);
    const bool ok = dst &&
                    cudaMemcpyAsync(dst, state, n * sizeof(float), cudaMemcpyDeviceToHost, s) ==
                        cudaSuccess &&
                    cudaStreamSynchronize(s) == cudaSuccess;
    ckpt_valid_ = ok ? grid : std::min(ckpt_valid_, grid - 1);
    // The ring slot held checkpoint grid - ckpt_n_. match() resumes at checkpoint c only while
    // hw_ - c * kGrid <= ring_ - window, and hw_ never falls within a sequence: older checkpoints are dead.
    // A segment stores its checkpoints before hw_ moves, hence the seg_ of margin.
    ckpt_free_ = std::max(ckpt_free_, grid - ckpt_n_);
    const int dead_below = std::max(hw_, grid * kGrid) - (ring_ - w_->cfg.window) - seg_;
    while ((ckpt_free_ + 1) * kGrid < dead_below)
        ++ckpt_free_;
}

int InklingForward::agree_(std::span<const int32_t> tokens, size_t lim) const {
    lim = std::min({hist_.size(), static_cast<size_t>(pos_), lim, tokens.size()});
    size_t d = 0;
    while (d < lim && hist_[d] == tokens[d])
        ++d;
    return static_cast<int>(d);
}

int InklingForward::match(std::span<const int32_t> tokens) const {
    // The last row is always computed: its logits start the reply.
    const int g = std::min(agree_(tokens, tokens.empty() ? 0 : tokens.size() - 1) / kGrid, ckpt_valid_);
    return g > 0 && held_(g) ? g * kGrid : 0;
}

int InklingForward::rewind(std::span<const int32_t> tokens, cudaStream_t s, int limit) {
    int g = std::min(match(tokens), std::max(limit, 0)) / kGrid;
    const size_t n = conv_state_floats_ * conv_state_.size();
    if (g > 0 && (!held_(g) || cudaMemcpyAsync(conv_all_, ckpt_at_(g), n * sizeof(float),
                                               cudaMemcpyHostToDevice, s) != cudaSuccess))
        g = 0;
    hist_.assign(tokens.begin(), tokens.end());
    armed_ = true;
    look_pos0_ = -1;  // the restored (or reset) conv state replaces a lookahead call's
    reset_draft_policy_();
    if (g == 0)
        return 0;  // run() at position 0 resets
    armed_ = false;
    pos_ = g * kGrid;
    ckpt_valid_ = g;
    if (spec_.restart)
        spec_.restart(pos_);
    return pos_;
}

void InklingForward::set_snapshot_draft(const InklingDraftConfig* d) {
    if (w_)
        snap_ = inkling_snap_layout(layout_, w_->cfg.window, d);
}

int InklingForward::snapshot_pos(std::span<const int32_t> tokens) const {
    const int g = std::min(agree_(tokens, tokens.size()) / kGrid, ckpt_valid_);
    return g > 0 && held_(g) ? g * kGrid : 0;
}

bool InklingForward::snapshot_pieces(int pos, std::vector<SnapshotPiece>& out, cudaStream_t s) {
    out.clear();
    if (pos <= 0 || pos % kGrid != 0 || !conv_all_ || !held_(pos / kGrid))
        return false;
    out.push_back({ckpt_at_(pos / kGrid), 0, conv_state_floats_ * conv_state_.size() * sizeof(float)});
    std::vector<InklingSnapCopy> rings;
    inkling_snap_ring_copies(layout_, w_->cfg.window, pos, rings);
    const std::byte* base = reinterpret_cast<const std::byte*>(conv_all_);
    for (const InklingSnapCopy& c : rings)
        out.push_back({base + c.slot_off, c.entry_off, c.bytes});
    // The draft's rows, or none: a restore of this entry into a slot with a draft must not read a window.
    if (snapshot_has_draft() && !spec_.save(pos, snap_.draft_off, out, s)) {
        out.clear();
        return false;
    }
    return true;
}

bool InklingForward::restore(std::span<const int32_t> tokens, const void* entry, int pos, cudaStream_t s,
                             bool with_draft) {
    const int g = pos / kGrid;
    if (!entry || !conv_all_ || ckpt_n_ < 1 || pos <= 0 || pos % kGrid != 0 || pos > max_ctx_ ||
        static_cast<size_t>(pos) >= tokens.size() || (spec_.load && !(with_draft && snap_.draft_bytes > 0)))
        return false;
    std::vector<InklingSnapCopy> rings;
    inkling_snap_ring_copies(layout_, w_->cfg.window, pos, rings);
    const std::byte* src = static_cast<const std::byte*>(entry);
    std::byte* base = reinterpret_cast<std::byte*>(conv_all_);
    bool ok = cudaMemcpyAsync(conv_all_, src, conv_state_floats_ * conv_state_.size() * sizeof(float),
                              cudaMemcpyDefault, s) == cudaSuccess;
    for (const InklingSnapCopy& c : rings)
        ok = ok && cudaMemcpyAsync(base + c.slot_off, src + c.entry_off, c.bytes, cudaMemcpyDefault, s) ==
                       cudaSuccess;
    // rewind() at a checkpoint of its own, every earlier checkpoint and ring row another sequence's; the
    // draft takes its window from the entry and keeps no row of the slot's last sequence.
    hist_.assign(tokens.begin(), tokens.end());
    armed_ = false;
    look_pos0_ = -1;
    reset_draft_policy_();
    pos_ = hw_ = pos;
    ckpt_valid_ = 0;
    ckpt_free_ = g - 1;
    ckpt_lo_ = g;
    if (spec_.load)
        ok = ok && spec_.load(pos, src, snap_.draft_off, s);
    if (ok)
        checkpoint_at_(g, s, conv_all_);
    if (ok && ckpt_valid_ == g)
        return true;
    reset(s);
    (void)cudaStreamSynchronize(s);  // the caller may drop entry once this returns
    return false;
}

bool InklingDraft::snapshot_pieces(int pos, size_t off, std::vector<SnapshotPiece>& out, cudaStream_t s) {
    const InklingDraftConfig& c = w_->cfg;
    if (!ok_ || !bound_ || pos <= 0)
        return false;
    flush_(s);
    // Every row before pos fed in this sequence and converted; a ring row of the window not written over.
    if (kept_pos0_ + kept_ < pos || (ring_ > 0 && hw_ > pos - c.window + ring_))
        return false;
    const size_t row = static_cast<size_t>(c.kv_heads) * c.head_dim * sizeof(half);
    std::vector<InklingSnapCopy> cp;
    for (int i = 0; i < c.layers; ++i) {
        if (!c.local[static_cast<size_t>(i)])
            continue;
        for (const half* ring : {k_[static_cast<size_t>(i)], v_[static_cast<size_t>(i)]}) {
            cp.clear();
            inkling_ring_window_copies(0, ring_, row, c.window, pos, off, cp);
            for (const InklingSnapCopy& x : cp)
                out.push_back({reinterpret_cast<const std::byte*>(ring) + x.slot_off, x.entry_off, x.bytes});
            off += static_cast<size_t>(c.window) * row;
        }
    }
    return true;
}

bool InklingDraft::load(int pos, const std::byte* entry, size_t off, cudaStream_t s) {
    const InklingDraftConfig& c = w_->cfg;
    if (!ok_ || !bound_ || pos <= 0 || !entry)
        return false;
    const size_t row = static_cast<size_t>(c.kv_heads) * c.head_dim * sizeof(half);
    std::vector<InklingSnapCopy> cp;
    bool ok = true;
    for (int i = 0; i < c.layers; ++i) {
        if (!c.local[static_cast<size_t>(i)])
            continue;
        for (half* ring : {k_[static_cast<size_t>(i)], v_[static_cast<size_t>(i)]}) {
            cp.clear();
            inkling_ring_window_copies(0, ring_, row, c.window, pos, off, cp);
            // A ring keeps every row twice (rows r and r + ring): the block reads its window in one stretch.
            for (const InklingSnapCopy& x : cp)
                for (size_t twice = 0; twice < (ring_ > 0 ? 2u : 1u); ++twice)
                    ok = ok &&
                         cudaMemcpyAsync(reinterpret_cast<std::byte*>(ring) + x.slot_off +
                                             twice * static_cast<size_t>(ring_) * row,
                                         entry + x.entry_off, x.bytes, cudaMemcpyDefault, s) == cudaSuccess;
            off += static_cast<size_t>(c.window) * row;
        }
    }
    // Nothing of the slot's last sequence is kept: the feature rows from pos on arrive with the prefill.
    kept_pos0_ = pos, kept_ = 0, done_ = 0;
    drafted_pos_ = -1;
    hw_ = pos;
    return ok;
}

}  // namespace imp
