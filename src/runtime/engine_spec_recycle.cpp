// Token-recycling table helpers of the speculative verify step (engine_spec_ngram.cpp), moved verbatim.
#include "model/model.h"
#include "runtime/engine.h"
#include "runtime/request.h"
#include "runtime/token_recycle_draft.h"

#include <algorithm>
#include <memory>

namespace imp {

TokenRecycleTable& Engine::spec_recycle_table_() {
    if (!spec_recycle_) {
        const auto& scfg = runtime_config_.speculative;
        spec_recycle_ = std::make_unique<TokenRecycleTable>(model_->config_.vocab_size,
                                                            std::max(1, scfg.recycle_slots));
    }
    return *spec_recycle_;
}

// Ingest this request's not-yet-seen tokens into the engine-scoped
// adjacency table: prompt bigrams once (spec_recycle_fed == 0), then the
// new output tokens (every emit path, verify accepts, loop bursts,
// plain steps, lands in output_tokens, so one cursor covers them all).
void Engine::spec_recycle_feed_(Request& req) {
    TokenRecycleTable& tr = spec_recycle_table_();
    const auto& out = req.output_tokens;
    if (req.spec_recycle_fed == 0) {
        const auto& in = req.input_tokens;
        for (size_t i = 1; i < in.size(); ++i)
            tr.observe_pair(in[i - 1], in[i]);
        if (!in.empty() && !out.empty())
            tr.observe_pair(in.back(), out.front());
    }
    for (int i = std::max(1, req.spec_recycle_fed); i < static_cast<int>(out.size()); ++i)
        tr.observe_pair(out[i - 1], out[i]);
    req.spec_recycle_fed = static_cast<int>(out.size());
}

}  // namespace imp
