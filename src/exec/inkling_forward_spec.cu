// Inkling2-Loop decode lookahead (inkling_forward.h): the draft's pay-off, the stored multi-row call and
// the conv state behind it.

#include "exec/inkling_forward.h"

#include "compute/gemm.h"
#include "compute/inkling_ops.h"
#include "core/logging.h"
#include "core/cuda_raii.h"
#include "core/tensor.h"
#include "memory/vram_allocator.h"
#include "quant/nvfp4_gemm.h"

#include <algorithm>
#include <chrono>

namespace imp {

using namespace inkling;

// Pay-off of the draft, called at every step that is not served from a stored row; true = ask it.
// Against a served one-row step of 5.6 ms (5.03 ms call + head and sampling) the draft call and its
// bookkeeping cost 0.23 (1.19 ms call), a draft row 0.16 (0.9 ms): a step pays while its confirmed rows
// exceed that (0.37 / 0.20 before the I8B32 projections and the captured draft: 199.3 -> 202.9 tok/s). Judged
// over a running mean after kProbe steps; a draft that does not pay rests for kRest tokens.
bool InklingForward::draft_due_() {
    constexpr float kDraftCost = 0.23f, kRowCost = 0.16f, kMix = 0.1f;
    constexpr int kProbe = 24, kRest = 96;
    if (drafted_) {
        const float rows = look_pos0_ >= 0 ? static_cast<float>(look_n_ - 1) : 0.0f;
        mean_hits_ += kMix * (static_cast<float>(look_hits_) - mean_hits_);
        mean_rows_ += kMix * (rows - mean_rows_);
        if (++tried_ >= kProbe && mean_hits_ < kDraftCost + kRowCost * mean_rows_) {
            pause_ = kRest;
            tried_ = 0;
        }
    }
    if (look_pos0_ >= 0)
        ++stats_.confirmed[look_hits_];
    look_hits_ = 0;
    const bool due = pause_ == 0 && pos_ + kGraphRows <= max_ctx_;
    if (pause_ > 0) {
        ++stats_.paused;
        if (--pause_ == 0)
            mean_hits_ = 1.0f, mean_rows_ = 2.0f;  // a fresh probe starts from a paying guess
    }
    return due;
}

bool InklingForward::enable_lookahead(InklingSpecHooks hooks, std::vector<int> taps, const half* embed) {
    const InklingConfig& c = w_->cfg;
    const size_t H = static_cast<size_t>(c.hidden), n = conv_state_floats_ * conv_state_.size();
    if (taps.size() < 2 || !embed || !ok_)
        return false;
    feat_ = F_(taps.size() * static_cast<size_t>(seg_) * H);
    conv_pre_ = F_(n);
    conv_tmp_ = F_(n);
    conv_log_ = F_(n / 3 * kGraphRows);
    look_h_ = Hf_(H * kGraphRows);
    d_look_ids_ = static_cast<int32_t*>(alloc_(kGraphRows * sizeof(int32_t)));
    if (!ok_)
        return false;
    taps_ = std::move(taps);
    embed_ = embed;
    spec_ = std::move(hooks);
    // The small-call graphs were captured without the feature planes and the conv log.
    for (int i = 0; i < kGraphRows; ++i) {
        decode_exec_[i].reset(nullptr);
        decode_graph_[i].reset(nullptr);
    }
    return true;
}

void InklingForward::conv_at_(int row, float* dst, cudaStream_t s) {
    const InklingConfig& c = w_->cfg;
    conv_state_at(conv_pre_, conv_log_, dst, row, static_cast<int>(conv_state_.size()),
                  c.n_kv_heads * c.head_dim, c.hidden, kGraphRows, s);
}

// The conv state holds every row of the stored lookahead call; put back the last confirmed one.
void InklingForward::settle_(cudaStream_t s) {
    if (look_pos0_ < 0)
        return;
    const int last = pos_ - 1 - look_pos0_;
    if (last < look_n_ - 1)
        conv_at_(last, conv_all_, s);
    look_pos0_ = -1;
}

bool InklingForward::step_spec_(half* h, int32_t tok, cudaStream_t s) {
    const size_t Hd = static_cast<size_t>(w_->cfg.hidden), hb = Hd * sizeof(half);
    const int pos = pos_;
    // A stored row for this position whose token is the one fed: serve it.
    const int j = look_pos0_ >= 0 ? pos - look_pos0_ : 0;
    if (j >= 1 && j < look_n_ && tok == look_tok_[j]) {
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(h, look_h_ + static_cast<size_t>(j) * Hd, hb, cudaMemcpyDeviceToDevice, s));
        ++pos_;
        ++stats_.accepted;
        ++look_hits_;
        if (pos_ % kGrid == 0) {
            conv_at_(j, conv_tmp_, s);
            checkpoint_(s, conv_tmp_);
        }
        return ok_;
    }
    drafted_ = draft_due_();
    settle_(s);
    int32_t drafts[kGraphRows] = {};
    int n = 0;
    if (drafted_) {
        const auto t0 = std::chrono::steady_clock::now();
        n = spec_.propose(tok, h, pos, drafts, s);
        stats_.propose_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++stats_.proposals;
    }
    n = std::clamp(n, 0, kGraphRows - 1);
    const int rows = n + 1;
    if (decode_graph_ok_ && !decode_exec_[rows - 1].get())
        capture_decode_(rows, s);
    if (!decode_graph_ok_ || !decode_exec_[rows - 1].get()) {
        IMP_LOG_ERROR("inkling: lookahead needs the %d-row graph", rows);
        return false;
    }
    // Rows: the anchor's embedding as given, then the drafts' from the table.
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(dec_h_, h, hb, cudaMemcpyDeviceToDevice, s));
    if (n > 0) {
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_look_ids_, drafts, static_cast<size_t>(n) * sizeof(int32_t),
                                           cudaMemcpyHostToDevice, s));
        embed_rows(embed_, d_look_ids_, mix_[1], n, static_cast<int>(Hd), s);
        f32_to_f16(mix_[1], dec_h_ + Hd, static_cast<size_t>(n) * Hd, s);
        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(conv_pre_, conv_all_,
                                           conv_state_floats_ * conv_state_.size() * sizeof(float),
                                           cudaMemcpyDeviceToDevice, s));
    }
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_pos_, &pos, sizeof(int), cudaMemcpyHostToDevice, s));
    if (cudaGraphLaunch(decode_exec_[rows - 1].get(), s) != cudaSuccess)
        return false;
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h, dec_h_, hb, cudaMemcpyDeviceToDevice, s));
    look_pos0_ = -1;
    if (n > 0) {
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(look_h_, dec_h_, hb * static_cast<size_t>(rows), cudaMemcpyDeviceToDevice, s));
        look_pos0_ = pos;
        look_n_ = rows;
        look_tok_[0] = tok;
        std::copy(drafts, drafts + n, look_tok_ + 1);
    }
    ++stats_.steps;
    stats_.drafted += n;
    if (stats_.steps % 512 == 0)
        IMP_LOG_INFO(
            "inkling: lookahead %ld steps (%ld paused), %ld of %ld draft rows confirmed (%.2f tokens "
            "per step), draft call %.2f ms, steps by confirmed rows 0..7: %ld %ld %ld %ld %ld %ld %ld %ld",
            stats_.steps, stats_.paused, stats_.accepted, stats_.drafted,
            1.0 + static_cast<double>(stats_.accepted) / static_cast<double>(stats_.steps),
            stats_.propose_ms / static_cast<double>(std::max(stats_.proposals, 1L)), stats_.confirmed[0],
            stats_.confirmed[1], stats_.confirmed[2], stats_.confirmed[3], stats_.confirmed[4],
            stats_.confirmed[5], stats_.confirmed[6], stats_.confirmed[7]);
    pos_ = pos + 1;
    hw_ = std::max(hw_, pos + rows);
    if (pos_ % kGrid == 0) {
        // After row 0 of a logged call, or simply the state of a one-row call.
        if (n > 0)
            conv_at_(0, conv_tmp_, s);
        checkpoint_(s, n > 0 ? conv_tmp_ : conv_all_);
    }
    spec_.on_rows(feat_, seg_, rows, pos, s);
    return ok_;
}

}  // namespace imp
