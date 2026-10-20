// engine_spec_sampled.cpp: speculative verify for sampled requests (speculative.sampled).
//
// Every draft source (suffix/n-gram, MTP, token recycling) proposes one token per row. Drawing each
// chunk row from the request's own distribution and accepting while the draw equals the draft is
// exact rejection sampling for such a draft: d is accepted with p(d), and a rejection emits a draw
// from p without d (Leviathan et al. 2023, q = one-hot). Row seeds follow the eager sampler's
// per-output-index seed (compute_step_seed), so a seeded request draws what it would without speculation.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"

namespace imp {

namespace {
bool greedy_request(const Request& req) { return req.temperature <= 0.0f || req.top_k == 1; }
}  // namespace

const char* Engine::spec_sampling_refusal_(const Request& req) const {
    if (greedy_request(req))
        return nullptr;
    if (!runtime_config_.speculative.sampled)
        return "sampling_not_greedy";
    // MoE verify chunks shift the drawn distribution (expert GEMMs, docs/LIMITATIONS.md). GDN hybrids
    // verify: Qwen3.8-27B MTP vs eager coin length t=-0.38. Qwen3-30B-A3B P(H) t=+3.96 / +2.91 (2 seed sets).
    if (model_->profile().is_moe)
        return "sampling_verify_numerics";
    // The row sampler holds at most SAMPLE_MAX_TOP_K candidates per row; the eager path sorts past it.
    if (sample_effective_top_k(req.top_k, model_->config().vocab_size) > SAMPLE_MAX_TOP_K)
        return "sampling_top_k_large";
    return nullptr;
}

bool Engine::spec_mc_refused_(const Request& req) const {
    const bool penalties = req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f ||
                           req.presence_penalty != 0.0f;
    return penalties || !greedy_request(req);
}

bool Engine::spec_arm_verify_sampling_(const Request& req) {
    if (greedy_request(req)) {
        executor_->set_verify_sampling(nullptr);
        return false;
    }
    VerifySampling s;
    s.temperature = req.temperature;
    s.top_p = req.top_p;
    s.min_p = req.min_p;
    s.typical_p = req.typical_p;
    s.top_k = sample_effective_top_k(req.top_k, model_->config().vocab_size);
    s.seed0 = engine_internal::compute_step_seed(req);
    executor_->set_verify_sampling(&s);
    return true;
}

}  // namespace imp
