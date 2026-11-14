#pragma once

#include "memory/plan.h"  // kMeasuredLibraryReserveBytes

// Internal helpers shared across engine_*.cpp translation units.
// Not part of any public API; included only by src/runtime/engine*.cpp.

#include "runtime/engine.h"
#include "runtime/request.h"
#include "compute/sampling.h"
#include "model/tokenizer.h"
#include "model/ngram_table.h"  // ngram_context_at
#include "model/model.h"
#include "runtime/batch.h"
#include "exec/executor.h"
#include "runtime/kv_fp8_scale_init.h"  // kv_cache.fp8_scale (warmup calibration)
#include "core/logging.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace imp::engine_internal {

// Free prefill metadata buffers when not using the pre-allocated pool.
// d_block_tables_swa may be nullptr; cudaFreeAsync ignores it, which is
// what non-SWA callers pass. Omitting it here leaked one buffer per
// successful chunk on SWA models (#1644).
inline void free_prefill_buffers(int32_t* d_token_ids, int* d_positions, int* d_block_tables,
                                 int* d_block_tables_swa, int* d_context_lens, cudaStream_t stream) {
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_token_ids, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_positions, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables_swa, stream));
    IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_context_lens, stream));
}

// Per-step sampler seed: request seed (or hash(id) ^ clock, fixed at the request's first draw) + output
// token count. Masked to 31 bits: samplers read a negative seed as unset and draw with the fixed 42u.
inline int compute_step_seed(const Request& req) {
    if (req.seed < 0 && !req.draw_base_set) {
        req.draw_base = static_cast<unsigned>(std::hash<int>{}(req.id) ^
                                              std::chrono::steady_clock::now().time_since_epoch().count());
        req.draw_base_set = true;
    }
    const unsigned base = req.seed >= 0 ? static_cast<unsigned>(req.seed) : req.draw_base;
    return static_cast<int>((base + static_cast<unsigned>(req.output_tokens.size())) & 0x7fffffffu);
}

[[nodiscard]] inline bool greedy_request(const Request& req) {
    return req.temperature <= 0.0f || req.top_k == 1;
}

// req's speculative-verify draw at its next output index (seed0 = compute_step_seed); greedy -> top_k 0.
inline VerifySampling verify_sampling_for(const Request& req, int vocab_size) {
    VerifySampling s;
    if (greedy_request(req)) {
        s.top_k = 0;
        return s;
    }
    s.temperature = req.temperature;
    s.top_p = req.top_p;
    s.min_p = req.min_p;
    s.typical_p = req.typical_p;
    s.top_k = sample_effective_top_k(req.top_k, vocab_size);
    s.seed0 = compute_step_seed(req);
    return s;
}

// Batched verify draws, T rows per request (engine_spec_sampled.cpp): a sampled request's row j draws with
// its params and the seed of output index size() + j, a greedy one's rows argmax (top_k 0). Empty: all
// greedy.
std::vector<VerifySampling> verify_row_draws(const std::vector<std::shared_ptr<Request>>& reqs, int T,
                                             int vocab_size);
// MTP draft draws (coupled MTP drafts), row lv * m + i: request reqs[bound[i]]'s draft level lv
// draws with its params and the seed of output index size() + lv, the seed of the verify row that checks
// it. A greedy request's rows take top_k 1. Empty: no bound request samples.
inline std::vector<VerifySampling> draft_row_draws(const std::vector<std::shared_ptr<Request>>& reqs,
                                                   const std::vector<int>& bound, int levels,
                                                   int vocab_size) {
    std::vector<VerifySampling> rows;
    const size_t m = bound.size();
    for (size_t i = 0; i < m; ++i) {
        const VerifySampling s = verify_sampling_for(*reqs[static_cast<size_t>(bound[i])], vocab_size);
        if (s.top_k < 1 || s.top_k > SAMPLE_MAX_TOP_K)
            continue;
        if (rows.empty())
            rows.assign(m * static_cast<size_t>(levels), VerifySampling{.top_k = 1});
        for (int lv = 0; lv < levels; ++lv) {
            VerifySampling& r = rows[(static_cast<size_t>(lv) * m) + i];
            r = s;
            r.seed0 = static_cast<int>((static_cast<unsigned>(s.seed0) + static_cast<unsigned>(lv)) &
                                       0x7fffffffu);
        }
    }
    return rows;
}

// Build a TokenLogprobInfo on the host from the PROCESSED logits: the row the
// sampler drew from, after penalties, logit_bias, bans, the constraint mask,
// min_p and typical_p (executor_sampling.cu applies them in place). vLLM
// calls this `processed_logprobs`; a masked token reads as probability 0
// (AUDIT_ARCH_2026 E-3, documented in docs/API.md).
inline TokenLogprobInfo build_logprob_info(const float* h_logits, int vocab_size, int32_t sampled_token,
                                           int top_logprobs, Tokenizer* tok) {
    LogprobResult lp_result;
    compute_logprobs_cpu(h_logits, vocab_size, sampled_token, top_logprobs, &lp_result);

    TokenLogprobInfo info;
    info.logprob = lp_result.sampled_logprob;
    info.text = tok ? tok->decode_token(sampled_token) : std::string{};  // no tokenizer: ids only
    info.top.reserve(lp_result.top.size());
    for (const auto& [tid, tlp] : lp_result.top) {
        info.top.push_back({tid, tlp, tok ? tok->decode_token(tid) : std::string{}});
    }
    return info;
}

// Ensure workspace 0 is active (used before prefill and after decode).
inline void ensure_prefill_workspace(GraphExecutor* executor) {
    if (executor->has_decode_workspace() && executor->active_workspace() != 0) {
        executor->use_workspace(0);
    }
}

// runtime.no_vision: drop the tower before the arena demand is read, so it reserves nothing.
inline void drop_vision_tower_if_disabled(Model& model, bool no_vision) {
    if (no_vision && model.vision_tower) {
        IMP_LOG_INFO("runtime.no_vision: vision tower dropped, image and video parts answer 400");
        model.vision_tower.reset();
    }
}

// Arena bytes the Gemma-4 audio tower + clip scratch take (0 without a tower); uploaded at warmup.
inline size_t audio_arena_bytes(Model& model) {
    return model.audio_tower ? AudioPipeline::demand_bytes(*model.audio_tower) : 0;
}

// A tower that fails to come up leaves the text model serving: input_audio then answers 400.
inline void init_audio_pipeline(AudioPipeline& audio, Model& model) {
    if (model.audio_tower && !audio.init(*model.audio_tower))
        IMP_LOG_WARN("Gemma-4 audio tower failed to initialise: continuing without audio input");
}

// runtime.no_audio: same, for the Gemma-4 audio tower.
inline void drop_audio_tower_if_disabled(Model& model, bool no_audio) {
    if (no_audio && model.audio_tower) {
        IMP_LOG_INFO("runtime.no_audio: audio tower dropped, input_audio parts answer 400");
        model.audio_tower.reset();
    }
}

// The library reserve the PLAN charges: imp.conf's vram.library_reserve_mb, or
// the measured default when unset. Shared because both the audit table and the
// warmup reporter need the same number, and duplicating the ternary at two call
// sites is how the two drift apart (AUDIT B41 is that drift, one level up).
inline size_t library_reserve_charge(int library_reserve_mb) {
    return library_reserve_mb < 0 ? kMeasuredLibraryReserveBytes
                                  : (static_cast<size_t>(library_reserve_mb) << 20);
}

// The library reserve this start measured: max(forward window, whole-init residual), the residual
// being device use minus the pool ledger. None without a forward window, and none when the residual
// is negative: the ledger then counts a pool twice and hides the libraries (#2347, 27B: -253 MiB).
inline std::optional<size_t> library_reserve_measurement(size_t forward_window, int64_t residual) {
    if (forward_window == SIZE_MAX || residual < 0)
        return std::nullopt;
    return std::max(forward_window, static_cast<size_t>(residual));
}

// PLE n-gram context of each decode-step sequence: its tokens at pos0-2, pos0-1 (pos0 = its
// first row), into `out` [n_seq][ctx_len]. nullptr without PLE or when rows do not split evenly.
inline const int32_t* ple_step_context(int ctx_len, int32_t eos,
                                       const std::vector<std::shared_ptr<Request>>& rows,
                                       const std::vector<int32_t>& positions, int total_tokens,
                                       std::vector<int32_t>& out) {
    const int n_seq = static_cast<int>(rows.size());
    if (ctx_len <= 0 || n_seq <= 0 || total_tokens % n_seq != 0 ||
        positions.size() < static_cast<size_t>(total_tokens))
        return nullptr;
    const int per_seq = total_tokens / n_seq;
    out.assign(static_cast<size_t>(n_seq) * ctx_len, 0);
    for (int s = 0; s < n_seq; s++) {
        const Request& r = *rows[static_cast<size_t>(s)];
        // false = positions past the history are eos-filled, the documented contract (ngram_table.h).
        (void)ngram_context_at(r.input_tokens.data(), static_cast<int>(r.input_tokens.size()),
                               r.output_tokens.data(), static_cast<int>(r.output_tokens.size()),
                               positions[static_cast<size_t>(s) * per_seq], ctx_len, eos,
                               out.data() + (static_cast<size_t>(s) * ctx_len));
    }
    return out.data();
}

// The decode step's host work (GraphExecutor::prepare_decode_step_host) with each row's PLE
// context. True = PLE rows staged for this step (InferenceState::ple_host_ready).
[[nodiscard]] inline bool prepare_decode_step_host(GraphExecutor& ex, const Model& model,
                                     const std::vector<std::shared_ptr<Request>>& rows, const Batch& batch,
                                     cudaStream_t stream) {
    const int32_t* ctx = ple_step_context(ex.ple_context_len(), model.config().ple_eos_token_id, rows,
                                          batch.positions, batch.total_tokens, ex.ngram_step_scratch());
    return ex.prepare_decode_step_host(batch.token_ids.data(), batch.total_tokens,
                                       static_cast<int>(rows.size()), ctx, stream);
}

// D2H copy, then stream sync; the first failure is returned.
inline cudaError_t copy_d2h_sync(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    const cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
    return err != cudaSuccess ? err : cudaStreamSynchronize(stream);
}

}  // namespace imp::engine_internal
