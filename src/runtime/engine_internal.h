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

// Per-step sampler seed: request seed (or hash(id) ^ clock) + output token count.
// Masked to 31 bits: samplers read a negative seed as unset and draw with the fixed 42u.
inline int compute_step_seed(const Request& req) {
    const unsigned base = req.seed >= 0 ? static_cast<unsigned>(req.seed)
                                        : static_cast<unsigned>(
                                              std::hash<int>{}(req.id) ^
                                              std::chrono::steady_clock::now().time_since_epoch().count());
    return static_cast<int>((base + static_cast<unsigned>(req.output_tokens.size())) & 0x7fffffffu);
}

// Build a TokenLogprobInfo on the host from the PROCESSED logits: the row the
// sampler drew from, after penalties, logit_bias, bans, the constraint mask,
// min_p and typical_p (executor_sampling.cu applies them in place). vLLM
// calls this `processed_logprobs`; a masked token reads as probability 0
// (AUDIT_arch_2026 E-3, documented in docs/API.md).
inline TokenLogprobInfo build_logprob_info(const float* h_logits, int vocab_size, int32_t sampled_token,
                                           int top_logprobs, Tokenizer* tok) {
    LogprobResult lp_result;
    compute_logprobs_cpu(h_logits, vocab_size, sampled_token, top_logprobs, &lp_result);

    TokenLogprobInfo info;
    info.logprob = lp_result.sampled_logprob;
    info.text = tok->decode_token(sampled_token);
    info.top.reserve(lp_result.top.size());
    for (const auto& [tid, tlp] : lp_result.top) {
        info.top.push_back({tid, tlp, tok->decode_token(tid)});
    }
    return info;
}

// Ensure workspace 0 is active (used before prefill and after decode).
inline void ensure_prefill_workspace(GraphExecutor* executor) {
    if (executor->has_decode_workspace() && executor->active_workspace() != 0) {
        executor->use_workspace(0);
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
                         out.data() + static_cast<size_t>(s) * ctx_len);
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

// Residual decode buffers, allocated ONCE (#1648): a captured forward_logits graph bakes their
// addresses. Failed alloc/init leaves a buffer unused (null, capacity 0 or an empty upload
// cache), which engine_scheduler.cpp skips; ~Engine frees whatever was allocated.
void alloc_residual_decode_buffers(int n, int*& d_slot, std::vector<int>& slot_uploaded, int*& d_meta,
                                   int& meta_cap);

// D2H copy, then stream sync; the first failure is returned.
inline cudaError_t copy_d2h_sync(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    const cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
    return err != cudaSuccess ? err : cudaStreamSynchronize(stream);
}

// Single-seq residual slot into d_buf[0], skipped while uploaded[0] already holds it.
// False: upload failed, cache cleared (retried next step), d_buf unusable this step.
[[nodiscard]] inline bool upload_residual_slot(int* d_buf, int slot, std::vector<int>& uploaded, cudaStream_t stream) {
    if (!uploaded.empty() && uploaded[0] == slot)
        return true;
    const cudaError_t err = cudaMemcpyAsync(d_buf, &slot, sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        uploaded.clear();
        IMP_LOG_ERROR("residual slot upload failed: %s", cudaGetErrorString(err));
        return false;
    }
    if (uploaded.empty())
        uploaded.assign(1, -1);
    uploaded[0] = slot;
    return true;
}

// Multi-seq residual metadata: slots, counts, write_idxes as [n] arrays at stride `cap` from
// base; the first failure is returned.
inline cudaError_t upload_residual_meta(int* base, ptrdiff_t cap, const int* slots, const int* counts,
                                        const int* widxes, int n, cudaStream_t stream) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(int);
    cudaError_t err = cudaMemcpyAsync(base, slots, bytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess)
        err = cudaMemcpyAsync(base + cap, counts, bytes, cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess)
        err = cudaMemcpyAsync(base + 2 * cap, widxes, bytes, cudaMemcpyHostToDevice, stream);
    return err;
}

}  // namespace imp::engine_internal
