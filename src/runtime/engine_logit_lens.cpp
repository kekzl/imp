// Logit lens (roadmap row 115): which rows of a step stash their residual, and the per-layer readout
// after the step sampled. Raw model distribution per layer: penalties, bias and constraints are not
// applied (the sampled token's logprob in output_logprobs is the processed one).

#include "runtime/engine.h"
#include "core/logging.h"
#include "exec/executor.h"
#include "exec/logit_lens.h"
#include "model/model.h"
#include "model/tokenizer.h"
#include "runtime/request.h"

#include <algorithm>

namespace imp {

bool Engine::logit_lens_supported() const {
    // Qwen4Exp's gated residual is hc_count x d_model wide, not the stream the final norm reads.
    return model_ && executor_ && !model_->profile().gated_residual;
}

bool Engine::decode_eager_only_(const std::vector<std::shared_ptr<Request>>& rows) {
    std::vector<int> idx;  // decode: hidden_ row i is rows[i]
    for (int i = 0; i < static_cast<int>(rows.size()); ++i)
        if (rows[i]->logit_lens > 0)
            idx.push_back(i);
    const bool lens = !idx.empty();
    if (lens)
        executor_->set_lens_rows(std::move(idx));
    return lens || runtime_config_.diagnostics.profile;
}

void Engine::logit_lens_arm_prefill_(const Request& req, int n_tokens) {
    if (req.logit_lens > 0 && n_tokens > 0)
        executor_->set_lens_rows({n_tokens - 1});  // the chunk's last row predicts the first output token
}

void Engine::logit_lens_collect_(const std::vector<std::shared_ptr<Request>>& rows,
                                 const std::vector<int32_t>& tokens, cudaStream_t stream) {
    std::vector<Request*> reqs;
    std::vector<int32_t> targets;
    int top_n = 0;
    for (size_t i = 0; i < rows.size() && i < tokens.size(); ++i) {
        if (rows[i]->logit_lens <= 0)
            continue;
        reqs.push_back(rows[i].get());
        targets.push_back(tokens[i]);
        top_n = std::max(top_n, rows[i]->logit_lens);
    }
    if (reqs.empty()) {
        executor_->set_lens_rows({});
        return;
    }
    LogitLensResult out;
    const bool ok = executor_->logit_lens_readout(targets, top_n, out, stream);
    executor_->set_lens_rows({});
    Tokenizer* tok = model_->tokenizer();
    for (int r = 0; r < static_cast<int>(reqs.size()); ++r) {
        // One entry per output token even on a failed readout: output_lens stays parallel to output_tokens.
        auto& step = reqs[r]->output_lens.emplace_back();
        if (!ok)
            continue;
        const int keep = std::min(reqs[r]->logit_lens, out.top_n);
        step.resize(out.layers);
        for (int l = 0; l < out.layers; ++l) {
            const size_t k = static_cast<size_t>(l) * out.rows + r;
            LogitLensLayer& ly = step[l];
            ly.logprob = out.lp[k];
            ly.rank = out.rank[k];
            ly.entropy = out.entropy[k];
            ly.top.reserve(keep);
            for (int j = 0; j < keep; ++j) {
                const int32_t id = out.top_ids[k * out.top_n + j];
                ly.top.push_back({id, out.top_lp[k * out.top_n + j], tok->decode_token(id)});
            }
        }
    }
    if (!ok)
        IMP_LOG_WARN("logit lens: readout failed for %zu row(s), entries left empty", reqs.size());
}

}  // namespace imp
