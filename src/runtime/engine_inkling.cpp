// Inkling2-Loop in the engine: what the first serving path of the arch runs without.

#include "core/logging.h"
#include "exec/executor.h"
#include "runtime/engine.h"
#include "runtime/request.h"

#include <algorithm>

namespace imp {

// One sequence, eager: exec/inkling_forward.h keeps one sequence's K/V and conv state outside the
// paged KV cache and reuses a prefix itself (inkling_resume_), so the engine's prefix cache is off.
// Context is capped at 32768 (8 KiB of global K/V per token).
void Engine::init_inkling_limits_() {
    if (model_->config().arch != ModelArch::INKLING2)
        return;
    constexpr int kInklingMaxCtx = 32768;
    config_.max_batch_size = 1;
    runtime_config_.runtime.max_batch_size = 1;
    if (scheduler_)
        scheduler_->clamp_max_batch_size(1);
    config_.use_cuda_graphs = false;
    config_.use_prefix_caching = false;
    runtime_config_.speculative.mtp_k = 0;
    runtime_config_.speculative.ngram = false;
    runtime_config_.speculative.capture = false;
    runtime_config_.runtime.prefill_graph = false;
    runtime_config_.runtime.prefill_batch = false;
    if (config_.max_seq_len <= 0 || config_.max_seq_len > kInklingMaxCtx)
        config_.max_seq_len = runtime_config_.runtime.max_seq_len > 0
                                  ? std::min(runtime_config_.runtime.max_seq_len, kInklingMaxCtx)
                                  : kInklingMaxCtx;
    runtime_config_.runtime.max_seq_len = config_.max_seq_len;
    IMP_LOG_INFO("inkling2_loop: one sequence, prefix reuse in the forward, no speculation, context %d",
                 config_.max_seq_len);
}

int Engine::inkling_resume_(Request& req, int offset, cudaStream_t stream) {
    if (offset != 0 || model_->config().arch != ModelArch::INKLING2 || !executor_)
        return offset;
    // Always called at offset 0: an empty span forgets the kept sequence.
    const bool reuse = req.prefix_reuse_allowed() && !ppl_capture_.active;
    const int r = executor_->inkling_resume(reuse ? req.key_tokens() : std::span<const int32_t>(), stream);
    if (r > 0) {
        IMP_LOG_INFO("inkling2_loop: seq %d resumes at %d of %zu prompt tokens", req.id, r,
                     req.input_tokens.size());
        req.prefill_offset = req.cached_tokens = r;
    }
    return r;
}

}  // namespace imp
