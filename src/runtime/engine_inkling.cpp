// Inkling2-Loop in the engine: what the first serving path of the arch runs without.

#include "core/logging.h"
#include "exec/executor.h"
#include "runtime/engine.h"
#include "runtime/request.h"

#include <algorithm>

namespace imp {

// Eager, one kept sequence per slot: exec/inkling_forward.h keeps K/V and conv state outside the
// paged KV cache and reuses a prefix itself (inkling_resume_), so the engine's prefix cache is off.
// Context is capped at 32768 (8 KiB of global K/V per token); a slot costs 2288 MiB, so 2 by
// default and 4 at most (inkling_fit_slots_ lowers it to what the card took).
void Engine::init_inkling_limits_() {
    if (model_->config().arch != ModelArch::INKLING2)
        return;
    constexpr int kInklingMaxCtx = 32768, kInklingMaxSlots = 4;
    const int slots = config_.max_batch_size > 0 ? std::min(config_.max_batch_size, kInklingMaxSlots) : 2;
    config_.max_batch_size = slots;
    runtime_config_.runtime.max_batch_size = slots;
    if (scheduler_)
        scheduler_->clamp_max_batch_size(slots);
    config_.use_cuda_graphs = false;
    config_.use_prefix_caching = false;
    runtime_config_.speculative.mtp_k = 0;
    runtime_config_.speculative.ngram = false;
    runtime_config_.speculative.capture = false;
    runtime_config_.runtime.prefill_graph = false;
    runtime_config_.runtime.prefill_batch = false;
    // Prefill runs at about 440 rows/s here and a decode step waits for the whole chunk: 8 rows per
    // waiter (x4 fairness = 32 for one ingest beside one stream) instead of 1024.
    runtime_config_.runtime.prefill_chunk_decode_cap =
        std::min(std::max(runtime_config_.runtime.prefill_chunk_decode_cap, 1), 8);
    if (config_.max_seq_len <= 0 || config_.max_seq_len > kInklingMaxCtx)
        config_.max_seq_len = runtime_config_.runtime.max_seq_len > 0
                                  ? std::min(runtime_config_.runtime.max_seq_len, kInklingMaxCtx)
                                  : kInklingMaxCtx;
    runtime_config_.runtime.max_seq_len = config_.max_seq_len;
    IMP_LOG_INFO("inkling2_loop: up to %d sequences, prefix reuse in the forward, no speculation, context %d",
                 slots, config_.max_seq_len);
}

void Engine::inkling_fit_slots_() {
    const int got = executor_ ? static_cast<int>(executor_->inkling_slot_requests().size()) : 0;
    if (got == 0 || got >= config_.max_batch_size)
        return;
    IMP_LOG_WARN("inkling2_loop: %d of %d sequence slots fit, batch size lowered", got,
                 config_.max_batch_size);
    config_.max_batch_size = got;
    runtime_config_.runtime.max_batch_size = got;
    if (scheduler_)
        scheduler_->clamp_max_batch_size(got);
}

int Engine::inkling_resume_(Request& req, int offset, cudaStream_t stream) {
    if (offset != 0 || model_->config().arch != ModelArch::INKLING2 || !executor_)
        return offset;
    // A slot is busy while the request bound to it is in this step's prefill or decode batch.
    const std::vector<int> bound = executor_->inkling_slot_requests();
    std::vector<bool> busy(bound.size());
    const auto live = [&](int id) {
        const auto is = [id](const std::shared_ptr<Request>& r) { return r && r->id == id; };
        return std::any_of(sched_prefill_batch_.begin(), sched_prefill_batch_.end(), is) ||
               std::any_of(sched_decode_batch_.begin(), sched_decode_batch_.end(), is);
    };
    for (size_t i = 0; i < bound.size(); ++i)
        busy[i] = bound[i] >= 0 && bound[i] != req.id && live(bound[i]);
    // Always called at offset 0: an empty span makes the slot forget its sequence.
    const bool reuse = req.prefix_reuse_allowed() && !ppl_capture_.active;
    const int r = executor_->inkling_resume(req.id, reuse ? req.key_tokens() : std::span<const int32_t>(),
                                            busy, stream);
    if (r > 0) {
        IMP_LOG_INFO("inkling2_loop: seq %d resumes at %d of %zu prompt tokens", req.id, r,
                     req.input_tokens.size());
        req.prefill_offset = req.cached_tokens = r;
    }
    return r;
}

}  // namespace imp
