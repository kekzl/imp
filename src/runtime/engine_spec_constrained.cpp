// engine_spec_constrained.cpp: speculative verify for greedy constrained requests (speculative.constrained).
//
// The masked argmax of a verify row equals its unmasked argmax exactly when the request's mask allows
// that token. step_spec_verify_ keeps a row only if spec_constraint_allows_ says so in the FSM state
// after the tokens emitted before it; the first disallowed row ends the chunk at the token before it.

#include "exec/executor_sampling_internal.h"
#include "exec/inference_state.h"
#include "runtime/constraint_manager.h"
#include "runtime/engine.h"
#include "runtime/request.h"

#include <cuda_runtime.h>

namespace imp {

namespace {
bool greedy(const Request& req) { return req.temperature <= 0.0f || req.top_k == 1; }
}  // namespace

bool Engine::spec_constrained_refused_(const Request& req) const {
    return req.wants_constraints() && !(runtime_config_.speculative.constrained && greedy(req));
}

bool Engine::spec_constraint_allows_(const Request& req, int32_t tok, cudaStream_t stream) {
    if (!req.constraints || !req.constraints->is_active())
        return true;
    const int vocab = model_->config().vocab_size;
    if (tok < 0 || tok >= vocab)
        return false;
    if (!spec_probe_row_) {
        spec_probe_row_ = VramOwned<char>(vram_alloc_, static_cast<size_t>(vocab) * sizeof(float),
                                          "spec_probe_row");
        if (!spec_probe_row_)
            return false;
    }
    auto* row = reinterpret_cast<float*>(spec_probe_row_.get());
    InferenceState st;
    st.json_constrainer = req.constraints->json_constrainer();
    st.schema_constrainer = req.constraints->schema_constrainer();
    st.regex_constrainer = req.constraints->regex_constrainer();
    st.grammar_constrainer = req.constraints->grammar_constrainer();
    st.constrain_remaining_tokens = req.max_tokens - static_cast<int>(req.output_tokens.size());
    if (cudaMemsetAsync(row, 0, static_cast<size_t>(vocab) * sizeof(float), stream) != cudaSuccess)
        return false;
    apply_constraint_mask(st, row, vocab, stream);
    float v = -1.0f;
    if (cudaMemcpyAsync(&v, row + tok, sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        return false;
    return v == 0.0f;  // masks write -FLT_MAX, never raise a logit
}

}  // namespace imp
