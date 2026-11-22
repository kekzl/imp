// Inkling2-Loop in the engine: what the first serving path of the arch runs without.

#include "core/logging.h"
#include "runtime/engine.h"

#include <algorithm>

namespace imp {

// One sequence, eager, no prefix reuse: exec/inkling_forward.h keeps one sequence's K/V and conv
// state outside the paged KV cache. Context is capped at 32768 (8 KiB of global K/V per token).
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
    IMP_LOG_INFO("inkling2_loop: one sequence, eager decode, no prefix cache, no speculation, context %d",
                 config_.max_seq_len);
}

}  // namespace imp
