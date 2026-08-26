// Pipelined batched decode (bd_pipe_): one decode step kept in flight
// so host bookkeeping and SSE delivery overlap GPU compute.
//
// Split out of engine_scheduler.cpp on 2026-08-26 (the file had doubled to
// 2230 code LOC past its allowlist rationale). Pure move: the function bodies
// are byte-identical to their previous form in that file.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/config.h"
#include "core/buffer.h"
#include "runtime/batch.h"
#include "compute/mtp_forward.h"
#include "compute/dispatch_record.h"
#include "memory/kv_cache.h"
#include "compute/sampling.h"
#include "core/logging.h"

#include <climits>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <vector>
#include <utility>

namespace imp {

using engine_internal::compute_step_seed;

// =====================================================================
// Pipelined batched decode (bd_pipe_) — one step in flight.
//
// Step N+1 is enqueued (device token chain feeding step N's sampled slot
// tokens as input ids + forward-graph replay + per-row sampler enqueue into
// the OTHER slot-parity half) BEFORE step N's tokens are read back; the
// host then event-waits on step N's gather only, so its bookkeeping + the
// server's SSE delivery overlap the GPU's work on N+1 instead of idling it
// (the ~15-20% step tail at n=16 sustained serving, nsys 2026-07-12).
//
// Scope (v1) is the clean serving case: every row async-sampleable (greedy
// or top-k<=SAMPLE_MAX_TOP_K/top-p; min_p/typical_p fine), no penalties/DRY
// (their token history grows per step host-side), no constraints/logprobs/
// mirostat/logit_bias, no SSM/SWA/StreamingLLM/residual-KV, CUDA graphs on.
// Anything else keeps the per-step path bit-for-bit unchanged.
//
// Known (accepted) semantics deltas vs the per-step path:
//  - host state lags the in-flight step by one token, so think-budget
//    forcing and stop-string detection fire one step later (one discarded
//    token; KV stays consistent because the row's release is deferred);
//  - a row that stops still has step N+1 computed for it — its token is
//    discarded and its KV blocks are freed only after that step completes
//    (deferred_release), so the in-flight forward never writes freed blocks.
// =====================================================================

namespace {
inline int pipeline_bucket_pow2(int x) {
    if (x <= 1)
        return 1;
    int b = 1;
    while (b < x)
        b <<= 1;
    return b;
}
}  // namespace

bool Engine::pipeline_row_eligible_(const Request& r) const {
    if (r.logprobs || r.json_mode || !r.json_schema.empty() || !r.regex_pattern.empty() ||
        !r.grammar.empty() || r.constraints)
        return false;
    if (r.mirostat != 0 || !r.logit_bias.empty())
        return false;
    // rep/freq/presence penalties ARE supported (device-side history — the
    // server defaults to repetition_penalty 1.05, so the common serving row
    // carries them). DRY needs the host-side token array — excluded.
    if (r.dry_multiplier > 0.0f)
        return false;
    const bool greedy = (r.temperature <= 0.0f || r.top_k == 1);
    if (!greedy) {
        const int vocab = model_ ? model_->config_.vocab_size : 0;
        const int top_k = r.top_k > 0 ? r.top_k : 50;
        const int eff_top_k = (top_k <= 0 || top_k > vocab) ? vocab : top_k;
        if (eff_top_k > SAMPLE_MAX_TOP_K)
            return false;  // CUB regime syncs internally — not enqueue-only
    }
    return true;
}

bool Engine::pipeline_batch_eligible_(const std::vector<std::shared_ptr<Request>>& rows) const {
    if (!runtime_config_.runtime.decode_pipeline)
        return false;
    const int n = static_cast<int>(rows.size());
    if (n < 2 || n > kMaxGraphPoolSize)
        return false;
    if (!config_.use_cuda_graphs || runtime_config_.diagnostics.profile)
        return false;
    if (!decode_batch_pool_.is_allocated())
        return false;
    if (offload_mgr_)
        return false;
    // Recurrent hybrids: excluded in #975 because a GDN decode step then
    // served one sequence. #1750's batched decode (stable device slot table)
    // made the pipeline RUNNABLE here — and measured SLOWER: alternating A/B
    // on Qwen3.8-27B at 32 streams, fresh server per arm, 2026-08-25:
    // pipeline ON 862-914 tok/s aggregate (median ~890) against OFF 940-953
    // (median ~945), non-overlapping. The chained advance + event waits cost
    // more on the hybrid step than the overlapped host gap returns. Keep the
    // exclusion as a measured verdict, not an inherited one.
    if (ssm_state_)
        return false;
    if (swa_sizing_active_ || config_.streaming_kv_enabled)
        return false;
    if (kv_manager_ && kv_manager_->residual_enabled())
        return false;
    if (!executor_->sample_pipeline_ready())
        return false;
    for (const auto& r : rows)
        if (!pipeline_row_eligible_(*r))
            return false;
    return true;
}

// Same logging blind spot the prefill-graph gates had (#1646): seven
// conditions gate the pipelined loop and none of them said anything, so a
// model that never pipelines looks exactly like one that does. Report the
// closed gate once per process, from the first batch that got as far as
// asking.
void Engine::log_pipeline_gate_once_(const std::vector<std::shared_ptr<Request>>& rows) {
    static bool logged = false;
    if (logged)
        return;
    logged = true;
    bool rows_ok = true;
    for (const auto& r : rows)
        if (!pipeline_row_eligible_(*r)) {
            rows_ok = false;
            break;
        }
    IMP_LOG_INFO(
        "decode-pipeline: not entering — cfg=%d n=%d graphs=%d profile=%d pool=%d offload=%d "
        "ssm_ok=%d swa=%d streaming_kv=%d residual=%d sampler_ready=%d rows_ok=%d",
        (int)runtime_config_.runtime.decode_pipeline, (int)rows.size(), (int)config_.use_cuda_graphs,
        (int)runtime_config_.diagnostics.profile, (int)decode_batch_pool_.is_allocated(),
        (int)(offload_mgr_ != nullptr),
        (int)!(ssm_state_ && !(runtime_config_.runtime.gdn_batched_decode && d_ssm_seq_slots_ != nullptr)),
        (int)swa_sizing_active_, (int)config_.streaming_kv_enabled,
        (int)(kv_manager_ && kv_manager_->residual_enabled()), (int)executor_->sample_pipeline_ready(),
        (int)rows_ok);
}

InferenceState Engine::pipeline_row_state_(Request& req, int row_idx) const {
    InferenceState per = bd_pipe_.base_state;
    fill_sampling_params(req, per);
    // The chained step samples one output AHEAD of the host-visible count —
    // match the seed the eager path would compute after processing the
    // in-flight step (compute_step_seed = base + output count).
    per.seed = compute_step_seed(req) + 1;
    per.penalty_tokens = nullptr;
    per.n_penalty_tokens = 0;
    per.host_penalty_tokens = nullptr;
    // Penalty rows read the per-row device history: host-known tokens were
    // uploaded at entry, and the advance kernel appended the in-flight
    // token at position `count` — the chained step penalizes count+1
    // tokens, exactly what the eager path would upload for that step.
    const bool needs_pen = (req.repetition_penalty != 1.0f || req.frequency_penalty != 0.0f ||
                            req.presence_penalty != 0.0f);
    if (needs_pen && d_pipe_hist_) {
        per.penalty_tokens = d_pipe_hist_ + static_cast<size_t>(row_idx) * pipe_hist_stride_;
        per.n_penalty_tokens = static_cast<int>(req.output_tokens.size()) + 1;
    }
    per.schema_constrainer = nullptr;
    per.json_constrainer = nullptr;
    per.regex_constrainer = nullptr;
    per.grammar_constrainer = nullptr;
    per.n_sequences = 1;
    return per;
}

bool Engine::pipeline_staging_ensure_() {
    if (bt_patch_cap_ == 0) {
        // Lazy mapped-pinned staging: at most one appended block per row per
        // step, so kMaxGraphPoolSize entries per parity set cover any batch.
        // The per-row device output-token history (penalty rows) is sized
        // to the model context (outputs are KV-bounded) with a sanity cap.
        const int msl = model_ ? model_->config().max_seq_len : 0;
        pipe_hist_stride_ = std::max(1024, std::min(msl > 0 ? msl : 32768, 65536));
        bool ok = true;
        // T5b, mapped: PinnedBuffer::device() IS the cudaHostGetDevicePointer
        // result, so the three separate lookups are gone with the three allocs
        // (memory/host_pinned.h).
        auto pin_mapped = [](PinnedBuffer& b, int** d_out, size_t bytes) {
            b = PinnedBuffer::acquire(cuda_host_pinned_allocator(), bytes, HostPinnedKind::Mapped);
            if (b.empty())
                return false;
            *d_out = b.device_as<int>();
            return true;
        };
        for (int p = 0; p < 2 && ok; ++p) {
            const size_t patch_bytes = kMaxGraphPoolSize * sizeof(int);
            ok = pin_mapped(h_bt_patch_off_[p], &d_bt_patch_off_[p], patch_bytes) &&
                 pin_mapped(h_bt_patch_val_[p], &d_bt_patch_val_[p], patch_bytes) &&
                 pin_mapped(h_hist_pos_[p], &d_hist_pos_[p], patch_bytes);
        }
        if (ok)
            ok = cudaMalloc(&d_pipe_hist_, static_cast<size_t>(kMaxGraphPoolSize) * pipe_hist_stride_ *
                                               sizeof(int32_t)) == cudaSuccess;
        if (!ok) {
            for (int p = 0; p < 2; ++p) {
                h_bt_patch_off_[p].reset();
                h_bt_patch_val_[p].reset();
                h_hist_pos_[p].reset();
                d_bt_patch_off_[p] = d_bt_patch_val_[p] = d_hist_pos_[p] = nullptr;
            }
            if (d_pipe_hist_) {
                cudaFree(d_pipe_hist_);
                d_pipe_hist_ = nullptr;
            }
            bt_patch_cap_ = -1;  // don't retry every step
            return false;
        }
        bt_patch_cap_ = kMaxGraphPoolSize;
    }
    return bt_patch_cap_ > 0;
}

int Engine::pipeline_prebook_kv_(int parity) {
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    const int stride = decode_batch_pool_.max_blocks_per_seq();
    if (!pipeline_staging_ensure_())
        return -1;

    int n_patches = 0;
    for (int i = 0; i < bd_pipe_.n; ++i) {
        auto& req = bd_pipe_.rows[i];
        const int target_ctx = req->context_len() + 1;
        const int blocks_needed = (target_ctx + kv_bs - 1) / kv_bs;
        const auto& bt = kv_manager_->block_table(req->id);
        const int have = static_cast<int>(bt.size());
        if (blocks_needed > have) {
            if (blocks_needed > stride)
                return -1;  // batch-pool stride exhausted — per-step path re-pads
            const int nb = kv_manager_->append_block(req->id);
            if (nb < 0)
                return -1;  // KV exhausted — per-step path handles reject/streaming
            h_bt_patch_off_[parity].as<int>()[n_patches] = i * stride + have;
            h_bt_patch_val_[parity].as<int>()[n_patches] = nb;
            ++n_patches;
        }
    }
    return n_patches;
}

bool Engine::pipeline_enqueue_next_(int parity, cudaStream_t stream) {
    const int n = bd_pipe_.n;
    if (n <= 0 || bd_pipe_.graph_idx < 0)
        return false;
    auto& runner = decode_graph_pool_[bd_pipe_.graph_idx];
    if (!runner.is_ready())
        return false;
    // #948 guard: the captured decode-attention launch topology covers the
    // pow2 context bucket of the capture — never chain past it (the per-step
    // path re-derives the graph at the boundary).
    int next_max_ctx = 0;
    for (const auto& r : bd_pipe_.rows)
        next_max_ctx = std::max(next_max_ctx, r->context_len() + 1);
    if (pipeline_bucket_pow2(next_max_ctx) > last_decode_max_ctx_per_graph_[bd_pipe_.graph_idx])
        return false;

    const int n_patches = pipeline_prebook_kv_(parity);
    if (n_patches < 0)
        return false;

    // Per-row history append positions (= host-visible output count; the
    // in-flight token lands there). Capacity check: a row whose history
    // would outgrow the stride drains to the per-step path.
    for (int i = 0; i < n; ++i) {
        const int count = static_cast<int>(bd_pipe_.rows[i]->output_tokens.size());
        if (count + 2 > pipe_hist_stride_)
            return false;
        h_hist_pos_[parity].as<int>()[i] = count;
    }

    // Device-side chain: feed the IN-FLIGHT step's sampled tokens (other
    // parity's slots) as this step's input ids, bump positions/context lens,
    // append them to the per-row history, scatter freshly appended block
    // ids. Stream-ordered after the in-flight step's samplers, before this
    // step's forward.
    decode_pipeline_advance(n, executor_->sample_slot_base(parity ^ 1), SAMPLE_SCRATCH_BYTES,
                            bd_pipe_.gpu.d_token_ids, bd_pipe_.gpu.d_positions, bd_pipe_.gpu.d_context_lens,
                            bd_pipe_.gpu.d_block_tables, n_patches, d_bt_patch_off_[parity],
                            d_bt_patch_val_[parity], d_pipe_hist_, pipe_hist_stride_, d_hist_pos_[parity],
                            stream);
    if (!runner.replay_only(stream))
        return false;  // abandoned chained forward is safe: the per-step path
                       // re-uploads batch state and rewrites the same KV slot

    executor_->set_sample_parity(parity);
    Tensor logits = executor_->get_logits_view(n);
    for (int i = 0; i < n; ++i) {
        InferenceState per = pipeline_row_state_(*bd_pipe_.rows[i], i);
        Tensor seq_logits = logits.slice(i, i + 1);
        if (!executor_->sample_single_from_logits_async(seq_logits, per, i, stream)) {
            // Statically unreachable (eligibility excludes every decline
            // mode); abandoning the half-enqueued step is safe (see above).
            IMP_LOG_ERROR("decode-pipeline: unexpected sampler decline (row %d) — not chaining", i);
            return false;
        }
    }
    return executor_->gather_sampled_tokens_async(n, stream);
}

bool Engine::pipeline_enter_(std::vector<std::shared_ptr<Request>>& rows, const GPUBatch& gpu_batch,
                             int graph_idx, const InferenceState& state, const Tensor& logits,
                             cudaStream_t stream, std::vector<int32_t>& tokens_out) {
    const int n = static_cast<int>(rows.size());
    // Penalty rows need the device-side history BEFORE any sampler runs —
    // without it the entry pass would sample unpenalized. Bail to the
    // legacy path (which uploads d_penalty_tokens_ per row) if staging is
    // unavailable or a row's history would not fit.
    if (!pipeline_staging_ensure_())
        return false;
    for (const auto& req : rows) {
        if (static_cast<int>(req->output_tokens.size()) + 2 > pipe_hist_stride_)
            return false;
    }
    executor_->set_sample_parity(0);
    for (int i = 0; i < n; ++i) {
        auto& req = rows[i];
        InferenceState per = state;
        fill_sampling_params(*req, per);
        per.seed = compute_step_seed(*req);
        per.penalty_tokens = nullptr;
        per.n_penalty_tokens = 0;
        per.host_penalty_tokens = nullptr;
        const bool needs_pen = (req->repetition_penalty != 1.0f || req->frequency_penalty != 0.0f ||
                                req->presence_penalty != 0.0f);
        const int count = static_cast<int>(req->output_tokens.size());
        if (needs_pen && count > 0) {
            // Seed the per-row device history with the host-known tokens;
            // chained steps keep it current via the advance-kernel append.
            // The async H2D from the (pageable) vector is safe: the gather
            // event below is synced before process_outputs can grow it.
            int32_t* row_hist = d_pipe_hist_ + static_cast<size_t>(i) * pipe_hist_stride_;
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(row_hist, req->output_tokens.data(), count * sizeof(int32_t),
                                               cudaMemcpyHostToDevice, stream));
            per.penalty_tokens = row_hist;
            per.n_penalty_tokens = count;
        }
        per.schema_constrainer = nullptr;
        per.json_constrainer = nullptr;
        per.regex_constrainer = nullptr;
        per.grammar_constrainer = nullptr;
        per.n_sequences = 1;
        Tensor seq_logits = logits.slice(i, i + 1);
        if (!executor_->sample_single_from_logits_async(seq_logits, per, i, stream)) {
            // Statically unreachable; the already-applied row filters are
            // idempotent under the eligibility gates, so the legacy collect
            // path can safely re-run every row.
            return false;
        }
    }
    if (!executor_->gather_sampled_tokens_async(n, stream)) {
        const int32_t* toks = executor_->collect_sampled_tokens(n, stream);
        if (!toks)
            return false;
        tokens_out.assign(toks, toks + n);
        return true;
    }

    // Chain step N+1 while step N's gather is in flight.
    bd_pipe_.rows = rows;
    bd_pipe_.n = n;
    bd_pipe_.graph_idx = graph_idx;
    bd_pipe_.gpu = gpu_batch;
    bd_pipe_.base_state = state;
    bd_pipe_.parity = 0;
    bd_pipe_.in_flight = false;
    bd_pipe_.steps_since_spec_yield = 0;
    if (pipeline_enqueue_next_(1, stream)) {
        bd_pipe_.in_flight = true;
        bd_pipe_.parity = 1;
    } else {
        // A partially attempted chain may have left the executor on the
        // other parity half — every non-pipelined caller expects parity 0.
        executor_->set_sample_parity(0);
        bd_pipe_.rows.clear();
        bd_pipe_.n = 0;
    }

    const int32_t* toks = executor_->wait_gathered_tokens(0);
    if (!toks) {
        // No event support — should not happen past sample_pipeline_ready.
        abandon_decode_pipeline();
        return false;
    }
    tokens_out.assign(toks, toks + n);
    return true;
}

void Engine::step_decode_pipeline_(cudaStream_t stream) {
    auto& db = sched_decode_batch_;
    // Continue only with the EXACT same composition in the same order (the
    // scheduler preserves admission order; any join/leave/cancel changes the
    // set) and only while the static gates still hold. Apply the same
    // max_batch_size cap the per-step path applies — at overload the rows
    // beyond the cap are not being stepped either way.
    size_t eff = db.size();
    const int max_bs_cap = runtime_config_.runtime.max_batch_size;
    if (max_bs_cap > 0 && eff > static_cast<size_t>(max_bs_cap))
        eff = static_cast<size_t>(max_bs_cap);
    bool cont = sched_prefill_batch_.empty() && eff == bd_pipe_.rows.size();
    if (cont) {
        for (size_t i = 0; i < eff; ++i) {
            if (db[i].get() != bd_pipe_.rows[i].get()) {
                cont = false;
                break;
            }
        }
    }
    if (cont) {
        for (const auto& r : bd_pipe_.rows) {
            if (r->status != RequestStatus::DECODING) {
                cont = false;
                break;
            }
        }
    }
    if (cont)
        cont = pipeline_batch_eligible_(bd_pipe_.rows);

    // #1003: yield the chain periodically so the plain path's round-robin
    // spec verify gets a turn — once in flight the pipeline otherwise chains
    // until the composition changes, and batch>1 speculation never fires
    // (observed verify_steps=1 per request). Fields-only candidate check;
    // the draft-depth economics (min_draft) run in the RR branch. The chain
    // re-engages on the following step via the normal plain-path entry.
    if (cont && runtime_config_.speculative.batch_rr && runtime_config_.speculative.ngram && !ssm_state_ &&
        ++bd_pipe_.steps_since_spec_yield >= spec_rr_yield_interval_) {
        bd_pipe_.steps_since_spec_yield = 0;
        for (const auto& r : bd_pipe_.rows) {
            if (r->status == RequestStatus::DECODING && spec_ngram_enabled_(*r) && !r->spec_ngram_given_up) {
                cont = false;
                break;
            }
        }
    }

    const int next_parity = bd_pipe_.parity ^ 1;
    const bool chained = cont && pipeline_enqueue_next_(next_parity, stream);

    // Collect + process the in-flight step (event wait ≈ 0 in steady state —
    // it finished while the host was in the scheduler/server since last step).
    pipeline_collect_process_(stream);

    if (chained) {
        bd_pipe_.parity = next_parity;
        // If every row just finished, the scheduler goes idle and nothing
        // would ever collect the chained step — drain it now (its tokens are
        // all for FINISHED rows and get discarded).
        bool any_decoding = false;
        for (const auto& r : bd_pipe_.rows) {
            if (r->status == RequestStatus::DECODING) {
                any_decoding = true;
                break;
            }
        }
        if (!any_decoding)
            drain_decode_pipeline();
    } else {
        // Pipeline ends. If a chained enqueue was attempted and partially
        // enqueued (advance kernel / replay), that work may still reference
        // the deferred rows' KV — order the release behind it.
        if (cont)
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        bd_pipe_.in_flight = false;
        pipeline_run_deferred_releases_();
        executor_->set_sample_parity(0);
        bd_pipe_.rows.clear();
        bd_pipe_.n = 0;
    }
}

void Engine::pipeline_collect_process_(cudaStream_t stream) {
    (void)stream;
    const int32_t* toks = executor_->wait_gathered_tokens(bd_pipe_.parity);
    if (!toks)
        return;
    for (int i = 0; i < bd_pipe_.n; ++i) {
        auto& req = bd_pipe_.rows[i];
        if (req->status != RequestStatus::DECODING)
            continue;  // finished/cancelled while this step was in flight — discard
        const int32_t next_token = toks[i];
        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);
        if (should_stop(*req, next_token) || static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            // The successor step (if chained) still writes this row's next
            // KV slot — defer the release until no step references it.
            req->status = RequestStatus::FINISHED;
            bd_pipe_.deferred_release.push_back(req);
        }
        kv_manager_->touch(req->id);
    }
}

void Engine::pipeline_run_deferred_releases_() {
    for (auto& req : bd_pipe_.deferred_release)
        finish_request_release_(req);
    bd_pipe_.deferred_release.clear();
}

void Engine::drain_decode_pipeline() {
    if (bd_pipe_.in_flight) {
        pipeline_collect_process_(decode_stream());
        bd_pipe_.in_flight = false;
    }
    pipeline_run_deferred_releases_();
    if (executor_)
        executor_->set_sample_parity(0);
    bd_pipe_.rows.clear();
    bd_pipe_.n = 0;
}

void Engine::abandon_decode_pipeline() {
    // Exception/teardown path: never wait on the device here.
    bd_pipe_.in_flight = false;
    pipeline_run_deferred_releases_();
    if (executor_)
        executor_->set_sample_parity(0);
    bd_pipe_.rows.clear();
    bd_pipe_.n = 0;
}

}  // namespace imp
