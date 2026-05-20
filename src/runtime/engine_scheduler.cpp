// Engine scheduler: per-step prefill + decode driver.
// The bulk of engine.cpp at ~1250 LOC across 13 methods.
//
// Top-level flow:
//   step() → step_schedule() → step_prefill() OR step_decode()
//
// Prefill chain:
//   step_prefill → resolve_prefill_chunk_size_ → supports_chunked_prefill_
//                → prefill_allocate_kv_blocks_ → prefill_upload_metadata_
//                → step_prefill_one (per chunk)
//
// Decode chain:
//   step_decode → decode_build_inference_state_ → step_decode_forward
//               → step_decode_process_outputs
//
// Async graph resume:
//   step_async_graph_resume — for the CUDA-graph-captured decode path.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap. This is the biggest single TU split in Phase 4.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/config.h"
#include "runtime/batch.h"
#include "runtime/mtp_forward.h"
#include "memory/kv_cache.h"
#include "compute/sampling.h"
#include "compute/layernorm.h"
#include "core/logging.h"

#include <cstring>
#include <algorithm>
#include <vector>

namespace imp {

using engine_internal::build_logprob_info;
using engine_internal::compute_step_seed;
using engine_internal::ensure_prefill_workspace;
using engine_internal::free_prefill_buffers;

// =====================================================================
// step() — main inference loop
// =====================================================================


// =====================================================================
// step() — main inference loop
// =====================================================================

bool Engine::step() {
    // Fast path: async conditional graph loop completed on GPU.
    int async_result = step_async_graph_resume();
    if (async_result == 1)
        return true;  // still running
    if (async_result == -1) {
        return scheduler_->has_pending() || scheduler_->active_count() > 0;
    }

    // Schedule prefill/decode batches and reconfigure green contexts.
    if (!step_schedule())
        return false;

    // Process prefill requests.
    if (!sched_prefill_batch_.empty()) {
        step_prefill(prefill_stream());
        ensure_prefill_workspace(executor_.get());
    }

    // Process decode requests (batched).
    if (!sched_decode_batch_.empty()) {
        step_decode(decode_stream());
        ensure_prefill_workspace(executor_.get());
    }

    return scheduler_->has_pending() || scheduler_->active_count() > 0;
}

// =====================================================================
// step_async_graph_resume — handle async conditional graph loop
// Returns: 0 = no graph active, 1 = still running, -1 = generation done
// =====================================================================

int Engine::step_async_graph_resume() {
    if (async_graph_runner_.is_setup() && async_graph_req_) {
        auto& req = async_graph_req_;

        if (async_pending_tokens_.empty() && async_pending_cursor_ == 0) {
            cudaStream_t dec_stream = decode_stream();
            async_pending_tokens_ = async_graph_runner_.wait_and_get_tokens(dec_stream);
        }

        int32_t token = -1;
        if (async_pending_cursor_ < static_cast<int>(async_pending_tokens_.size())) {
            token = async_pending_tokens_[async_pending_cursor_++];
        }

        bool generation_done = false;
        if (token >= 0) {
            req->output_tokens.push_back(token);
            track_think_state(*req, token);
            bool is_stop = should_stop(*req, token);
            generation_done = is_stop || static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (!generation_done)
                return 1;
        }

        auto saved_req = async_graph_req_;

        async_graph_runner_.cleanup();
        if (async_d_block_tables_) {
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_));
            async_d_block_tables_ = nullptr;
        }
        if (async_d_banned_tokens_) {
            IMP_CUDA_CHECK_LOG(cudaFree(async_d_banned_tokens_));
            async_d_banned_tokens_ = nullptr;
        }
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;

        if (generation_done) {
            finish_request(saved_req);
            return -1;
        }

        IMP_LOG_DEBUG("AsyncGraphLoop: graph tokens exhausted, continuing with step decode");
    }

    // Clean up stale async graph state
    if (async_graph_req_ && !async_graph_runner_.is_setup()) {
        async_graph_req_ = nullptr;
        async_pending_tokens_.clear();
        async_pending_cursor_ = 0;
    }

    return 0;
}

// =====================================================================
// step_schedule — call scheduler, reconfigure green contexts
// Returns true if there is work to do.
// =====================================================================

bool Engine::step_schedule() {
    sched_prefill_batch_.clear();
    sched_decode_batch_.clear();
    scheduler_->schedule(sched_prefill_batch_, sched_decode_batch_);

    if (sched_prefill_batch_.empty() && sched_decode_batch_.empty()) {
        return false;
    }

    // Dynamic Green Context SM reconfiguration
    if (config_.use_green_contexts && green_ctx_.is_available() && green_ctx_.has_green_contexts()) {
        float target_ratio = config_.green_ctx_prefill_ratio;
        if (sched_prefill_batch_.empty() && !sched_decode_batch_.empty()) {
            target_ratio = 0.0f;
        } else if (!sched_prefill_batch_.empty() && sched_decode_batch_.empty()) {
            target_ratio = 1.0f;
        }
        if (std::abs(target_ratio - green_ctx_.prefill_ratio()) > 0.1f) {
            green_ctx_.reconfigure(target_ratio);
        }
    }

    return true;
}

// =====================================================================
// supports_chunked_prefill_ / resolve_prefill_chunk_size_
// Whether the model arch + KV dtype combination supports chunked prefill.
// Returns true for full-attention models (Qwen3, Llama, Mistral) and hybrid
// GDN+MoE / Mamba2+MoE models (Qwen3.5/3.6, Nemotron-H) with FP16, FP8, or
// NVFP4 KV cache. Returns false for SWA models (Gemma-3/4, Llama-4) and
// sub-byte KV dtypes (INT4, TurboQuant) lacking gather kernels.
// =====================================================================

bool Engine::supports_chunked_prefill_() const {
    if (!model_)
        return false;
    const auto& cfg = model_->config();
    // Out-of-scope archs. Hybrid GDN+MoE / Mamba2+MoE archs (QWEN35*,
    // QWEN36_MOE, NEMOTRON_H_MOE) ARE supported. Gemma-4 (SWA + dual
    // head_dim 256/512) is now supported via:
    //   - cuBLAS softmax sliding_window param (PR feat(attn): sliding_window),
    //   - per-layer dispatch through the rectangular cuBLAS prefill path
    //     (every layer call uses its own nh/nkv/hd from layer-local vars).
    if (cfg.arch == ModelArch::GEMMA3) return false;       // SWA, no test model
    if (cfg.arch == ModelArch::LLAMA4) return false;       // MoE + SWA, untested
    // Per-layer attention shape uniformity gate. Hybrid archs (QWEN35*, QWEN36_MOE,
    // NEMOTRON_H_MOE) populate n_kv_heads_per_layer with zeros for non-attention
    // layers — uniformity here means all *nonzero* values agree. Truly heterogeneous
    // shapes (Gemma-4 dual head_dim 256/512) are now allowed because the chunked
    // path dispatches per-layer with the correct nh/nkv/hd from layer-local vars.
    auto first_nonzero_int = [](const std::vector<int>& v) -> int {
        for (int x : v) if (x > 0) return x;
        return 0;
    };
    auto any_nonzero_differs = [](const std::vector<int>& v, int ref) -> bool {
        for (int x : v) if (x > 0 && x != ref) return true;
        return false;
    };
    if (!cfg.n_kv_heads_per_layer.empty()) {
        int ref = first_nonzero_int(cfg.n_kv_heads_per_layer);
        if (ref > 0 && any_nonzero_differs(cfg.n_kv_heads_per_layer, ref)) {
            // Allow Gemma-4: per-layer dispatch covers heterogeneous shapes.
            if (cfg.arch != ModelArch::GEMMA4) return false;
        }
    }
    if (!cfg.head_dim_per_layer.empty()) {
        int ref = first_nonzero_int(cfg.head_dim_per_layer);
        if (ref > 0 && any_nonzero_differs(cfg.head_dim_per_layer, ref)) {
            if (cfg.arch != ModelArch::GEMMA4) return false;
        }
    }
    // KV dtypes wired through paged_kv_gather: FP16, FP8_E4M3, NVFP4, MXFP4_KV, INT4.
    // INT8 and TurboQuant variants would need their own gather kernels.
    if (kv_cache_raw_) {
        QType kvt = kv_cache_raw_->qtype();
        if (kvt != QType::F16 && kvt != QType::FP8_E4M3 &&
            kvt != QType::NVFP4 && kvt != QType::MXFP4_KV && kvt != QType::INT4)
            return false;
    }
    return true;
}

int Engine::resolve_prefill_chunk_size_() const {
    int explicit_val = config_.prefill_chunk_size;
    if (explicit_val < 0) {
        return supports_chunked_prefill_() ? 512 : 0;
    }
    if (explicit_val == 0)
        return 0;
    // explicit_val > 0
    if (!supports_chunked_prefill_()) {
        IMP_LOG_WARN(
            "prefill_chunk_size=%d ignored: arch=%d / kv_dtype=%d not in chunked-prefill scope; using 0",
            explicit_val, (int)model_->config().arch,
            kv_cache_raw_ ? (int)kv_cache_raw_->qtype() : -1);
        return 0;
    }
    return explicit_val;
}

// =====================================================================
// step_prefill — process all prefill requests
// =====================================================================

void Engine::step_prefill(cudaStream_t stream) {
    int resolved    = resolve_prefill_chunk_size_();
    int effective_chunk = (resolved > 0) ? resolved : executor_->max_tokens();
    // Hard cap: chunk size must never exceed the executor's max_tokens
    // (which is itself capped to 256 for SSM/GDN+MoE hybrids and 512 for
    // dense GDN to bound workspace VRAM). Without this clamp, a server-side
    // prefill_chunk_size default of 512 (handlers.cpp) overflows the
    // workspace and crashes with `n_tokens (X) exceeds max_tokens (Y)` →
    // `terminate: reshape: numel mismatch` on long prompts to e.g. Qwen3.6.
    if (effective_chunk > executor_->max_tokens()) {
        effective_chunk = executor_->max_tokens();
    }
    if (kv_manager_) {
        int bs = kv_manager_->kv_cache()->block_size();
        if (effective_chunk > bs)
            effective_chunk = (effective_chunk / bs) * bs;
    }

    for (auto& req : sched_prefill_batch_) {
        step_prefill_one(req, effective_chunk, stream);
        kv_manager_->touch(req->id);
    }
}

// =====================================================================
// step_prefill_one — process a single prefill request
// =====================================================================

// Allocate KV blocks for a prefill step. Two sub-paths:
//   - prefix caching: try allocate_blocks_with_prefix, evict + retry on
//     budget pressure, advance `offset` past the reused prefix.
//   - plain: allocate `additional` blocks, evict + retry, cancel on hard
//     failure.
// Returns false on unrecoverable failure (req->status already set to
// CANCELLED). On prefix-cache reuse, mutates offset / chunk_len /
// is_last_chunk / ctx_len in place.
bool Engine::prefill_allocate_kv_blocks_(std::shared_ptr<Request>& req, int kv_bs,
                                         int total_input, int effective_chunk,
                                         int& offset, int& chunk_len, bool& is_last_chunk,
                                         int& ctx_len, cudaStream_t pf_stream) {
    int num_blocks = (ctx_len + kv_bs - 1) / kv_bs;
    int prefix_reused = 0;
    int existing = static_cast<int>(kv_manager_->block_table(req->id).size());

    if (kv_manager_->prefix_caching_enabled() && existing == 0 && offset == 0) {
        int total_blocks_needed = (total_input + kv_bs - 1) / kv_bs;
        prefix_reused = kv_manager_->allocate_blocks_with_prefix(req->id, req->input_tokens);
        if (prefix_reused < 0) {
            while (kv_manager_->num_free_blocks() < total_blocks_needed) {
                int evicted = kv_manager_->evict_lru();
                if (evicted < 0)
                    break;
            }
            prefix_reused = kv_manager_->allocate_blocks_with_prefix(req->id, req->input_tokens);
            if (prefix_reused < 0) {
                req->status = RequestStatus::CANCELLED;
                return false;
            }
        }

        if (prefix_reused > 0) {
            int effective_reused = (prefix_reused > 1) ? prefix_reused - 1 : 0;
            int skip_tokens = effective_reused * kv_bs;
            if (skip_tokens >= total_input) {
                skip_tokens = (total_input / kv_bs) * kv_bs;
                if (skip_tokens >= total_input) {
                    skip_tokens = total_input - 1;
                }
            }
            if (skip_tokens > offset) {
                IMP_LOG_INFO("PrefixCache: seq %d skipping %d/%d prefill tokens (%d blocks reused)",
                             req->id, skip_tokens, total_input, prefix_reused);
                req->cached_tokens = skip_tokens;
                offset = skip_tokens;
                req->prefill_offset = offset;
                chunk_len = total_input - offset;
                is_last_chunk = true;
                if (chunk_len > effective_chunk) {
                    chunk_len = effective_chunk;
                    is_last_chunk = false;
                }
                ctx_len = offset + chunk_len;
                (void)executor_->resize_workspace(chunk_len, pf_stream);
            }
        }
    } else {
        int additional = num_blocks - existing;
        if (additional > 0) {
            if (!kv_manager_->allocate_blocks(req->id, additional)) {
                while (kv_manager_->num_free_blocks() < additional) {
                    int evicted = kv_manager_->evict_lru();
                    if (evicted < 0)
                        break;
                }
                if (!kv_manager_->allocate_blocks(req->id, additional)) {
                    kv_manager_->free_sequence(req->id);
                    req->status = RequestStatus::CANCELLED;
                    return false;
                }
            }
        }
    }
    return true;
}

// Upload prefill metadata to device. Uses the prefill_pool_ pre-allocated
// buffers when chunk_len fits; otherwise falls back to cudaMallocAsync and
// frees on any allocation failure. Pinned staging buffers are used for the
// token_ids / positions H2D copies when available (avoids internal
// pageable→pinned copy inside cuMemcpy).
bool Engine::prefill_upload_metadata_(std::shared_ptr<Request>& req,
                                      const std::vector<int>& block_table,
                                      int chunk_len, int offset, int ctx_len,
                                      cudaStream_t pf_stream,
                                      int32_t*& d_token_ids, int*& d_positions,
                                      int*& d_block_tables, int*& d_context_lens,
                                      bool& pf_pool_used) {
    d_token_ids = nullptr;
    d_positions = nullptr;
    d_block_tables = nullptr;
    d_context_lens = nullptr;
    pf_pool_used = false;

    auto check = [&req](cudaError_t err, const char* op) {
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Engine::step prefill %s failed: %s", op, cudaGetErrorString(err));
            req->status = RequestStatus::CANCELLED;
        }
        return err == cudaSuccess;
    };

    if (prefill_pool_ && chunk_len <= config_.max_seq_len) {
        d_token_ids = d_pf_token_ids_;
        d_positions = d_pf_positions_;
        d_block_tables = d_pf_block_tables_;
        d_context_lens = d_pf_context_lens_;
        pf_pool_used = true;
    } else {
        if (!check(cudaMallocAsync(&d_token_ids, chunk_len * sizeof(int32_t), pf_stream),
                   "malloc token_ids") ||
            !check(cudaMallocAsync(&d_positions, chunk_len * sizeof(int), pf_stream), "malloc positions") ||
            !check(cudaMallocAsync(&d_block_tables, block_table.size() * sizeof(int), pf_stream),
                   "malloc block_tables") ||
            !check(cudaMallocAsync(&d_context_lens, sizeof(int), pf_stream), "malloc context_lens")) {
            if (d_token_ids)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_token_ids, pf_stream));
            if (d_positions)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_positions, pf_stream));
            if (d_block_tables)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_block_tables, pf_stream));
            if (d_context_lens)
                IMP_CUDA_CHECK_LOG(cudaFreeAsync(d_context_lens, pf_stream));
            kv_manager_->free_sequence(req->id);
            return false;
        }
    }

    // Use pinned staging buffers when available (avoids internal pageable->pinned copy)
    if (h_pf_token_ids_ && chunk_len <= config_.max_seq_len) {
        memcpy(h_pf_token_ids_, req->input_tokens.data() + offset, chunk_len * sizeof(int32_t));
        check(cudaMemcpyAsync(d_token_ids, h_pf_token_ids_, chunk_len * sizeof(int32_t),
                              cudaMemcpyHostToDevice, pf_stream),
              "memcpy token_ids");
    } else {
        check(cudaMemcpyAsync(d_token_ids, req->input_tokens.data() + offset, chunk_len * sizeof(int32_t),
                              cudaMemcpyHostToDevice, pf_stream),
              "memcpy token_ids");
    }

    if (h_pf_positions_ && chunk_len <= config_.max_seq_len) {
        for (int i = 0; i < chunk_len; i++)
            h_pf_positions_[i] = offset + i;
        check(cudaMemcpyAsync(d_positions, h_pf_positions_, chunk_len * sizeof(int), cudaMemcpyHostToDevice,
                              pf_stream),
              "memcpy positions");
    } else {
        std::vector<int> positions(chunk_len);
        for (int i = 0; i < chunk_len; i++)
            positions[i] = offset + i;
        check(cudaMemcpyAsync(d_positions, positions.data(), chunk_len * sizeof(int), cudaMemcpyHostToDevice,
                              pf_stream),
              "memcpy positions");
    }

    check(cudaMemcpyAsync(d_block_tables, block_table.data(), block_table.size() * sizeof(int),
                          cudaMemcpyHostToDevice, pf_stream),
          "memcpy block_tables");
    check(cudaMemcpyAsync(d_context_lens, &ctx_len, sizeof(int), cudaMemcpyHostToDevice, pf_stream),
          "memcpy context_lens");
    return true;
}

void Engine::step_prefill_one(std::shared_ptr<Request>& req, int effective_chunk, cudaStream_t pf_stream) {
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;
    int total_input = static_cast<int>(req->input_tokens.size());
    int offset = req->prefill_offset;

    // Out-of-scope archs (Gemma-3/4 SWA, Llama-4, sub-byte KV) lack a paged
    // chunked-prefill path, so the chunked-prefill branch in
    // executor_attention.cu aborts on chunk 2+ (q_offset > 0 + per_layer
    // shapes). Reject prompts > effective_chunk gracefully here instead of
    // letting them hit std::abort. Real fix is the paged hybrid-prefill
    // kernel (roadmap).
    if (offset == 0 && total_input > effective_chunk && !supports_chunked_prefill_()) {
        IMP_LOG_ERROR(
            "Prompt has %d tokens but max_tokens=%d on hybrid/out-of-scope arch — "
            "chunked prefill not supported. Cancelling request %d.",
            total_input, effective_chunk, req->id);
        req->status = RequestStatus::CANCELLED;
        return;
    }

    // Clamp effective_chunk so n × ctx_len ≤ s_cap² where s_cap is the
    // attn_scores_ workspace dimension. Worst case is the final chunk where
    // ctx_len ≈ total_input. Without this, long prompts on hybrid models
    // (Qwen3.5/3.6 GDN, Nemotron-H) hit "attn_scores_ capacity too small"
    // abort in executor_attention.cu — see chunked_prefill_attn_scores_capacity_bug.
    if (executor_) {
        int s_cap = executor_->attn_scores_cap();
        if (s_cap > 0 && total_input > 0) {
            int64_t cap2 = static_cast<int64_t>(s_cap) * s_cap;
            int max_chunk_for_buf = static_cast<int>(cap2 / total_input);
            max_chunk_for_buf = (max_chunk_for_buf / kv_bs) * kv_bs;
            if (max_chunk_for_buf > 0 && effective_chunk > max_chunk_for_buf) {
                effective_chunk = max_chunk_for_buf;
            }
        }
    }

    // Determine chunk boundaries
    int chunk_len = total_input - offset;
    bool is_last_chunk = true;
    if (chunk_len > effective_chunk) {
        chunk_len = effective_chunk;
        is_last_chunk = false;
    }

    int ctx_len = offset + chunk_len;
    (void)executor_->resize_workspace(chunk_len, pf_stream);

    if (!prefill_allocate_kv_blocks_(req, kv_bs, total_input, effective_chunk, offset, chunk_len,
                                     is_last_chunk, ctx_len, pf_stream)) {
        return;  // caller already set req->status = CANCELLED
    }

    const auto& block_table = kv_manager_->block_table(req->id);

    int32_t* d_token_ids = nullptr;
    int* d_positions = nullptr;
    int* d_block_tables = nullptr;
    int* d_context_lens = nullptr;
    bool pf_pool_used = false;
    if (!prefill_upload_metadata_(req, block_table, chunk_len, offset, ctx_len, pf_stream,
                                  d_token_ids, d_positions, d_block_tables, d_context_lens,
                                  pf_pool_used)) {
        return;  // caller already set req->status = CANCELLED
    }

    // Build InferenceState
    InferenceState state;
    state.token_ids = d_token_ids;
    state.positions = d_positions;
    state.n_tokens = chunk_len;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = d_block_tables;
    state.context_lens = d_context_lens;
    state.max_context_len = ctx_len;
    state.n_sequences = 1;
    state.max_blocks_per_seq = 0;
    state.is_prefill = true;
    state.prefill_offset = offset;  // absolute pos of state.positions[0]
    state.kv_manager = kv_manager_.get();
    if (kv_manager_ && kv_manager_->residual_enabled()) {
        // Slot lookup happens inside KVCacheManager::residual_k_ptr; if no
        // slot is allocated yet (prefill before first decode), the residual
        // pointers return nullptr and the kernel skips the residual pass.
        state.kv_seq_id = req->id;
    }
    fill_sampling_params(*req, state);

    // Constraints via ConstraintManager
    constraints_.prepare(req->json_mode, req->json_schema, model_->tokenizer(), req->has_tools,
                         req->tpl_family);
    state.json_constrainer = constraints_.json_constrainer();
    state.schema_constrainer = constraints_.schema_constrainer();

    // Penalties
    upload_penalties(*req, state, pf_stream);

    // Recurrent state (SSM/GDN)
    // Reset on the first chunk of a new request so previous-request state
    // doesn't leak in.  Subsequent chunks must NOT reset — the recurrent
    // state built during earlier chunks must carry forward.
    fill_recurrent_state(*req, state, /*reset=*/(offset == 0), pf_stream);

    // Vision embeddings on first chunk
    if (vision_.has_input() && vision_.is_available() && offset == 0) {
        state.vision_embeddings = vision_.embeddings();
        state.vision_token_id = vision_.soft_token_id();
        state.n_vision_tokens = vision_.num_image_tokens();
    }

    if (!is_last_chunk) {
        if (executor_->has_decode_workspace()) {
            executor_->use_workspace(0);
        }
        Tensor logits_out;

        // Prefill graph capture (opt-in, Phase 4 of MoE-prefill-graphs work).
        // Conditions: env-gated, pool path (stable device buffers), and
        // chunk shape stable (in practice all non-last chunks share chunk_len
        // = prefill_chunk_size). H2D upload happened above on pf_stream
        // *before* this wrapper — captured region is forward_logits only,
        // analogous to the decode graph pattern.
        const bool prefill_graph_enabled = RuntimeConfig::current().runtime.prefill_graph;
        const bool can_capture = prefill_graph_enabled && pf_pool_used && config_.use_cuda_graphs;
        if (can_capture) {
            const int block_count = static_cast<int>(block_table.size());
            if (chunk_len != last_prefill_chunk_len_ || block_count != last_prefill_block_count_) {
                prefill_graph_runner_.invalidate_for_update();
                last_prefill_chunk_len_ = chunk_len;
                last_prefill_block_count_ = block_count;
            }
            prefill_graph_runner_.set_decode_fn([this, &state, &logits_out](cudaStream_t s) {
                executor_->forward_logits(state, logits_out, s);
            });
            prefill_graph_runner_.execute(pf_stream);
            if (logits_out.data == nullptr) {
                logits_out = executor_->get_logits_view(/*n_sequences=*/1);
            }
        } else {
            executor_->forward_logits(state, logits_out, pf_stream);
        }

        if (!pf_pool_used) {
            free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
        }

        req->prefill_offset = offset + chunk_len;
        IMP_LOG_DEBUG("Chunked prefill: req %d chunk [%d, %d) of %d", req->id, offset, offset + chunk_len,
                      total_input);
    } else {
        // Last chunk: forward + sample
        int32_t next_token;
        bool use_event_sync = (h_sample_pinned_ != nullptr && executor_->d_sample_result() != nullptr &&
                               (state.temperature <= 0.0f || state.top_k == 1) && !req->logprobs &&
                               !state.json_constrainer && !state.schema_constrainer);

        Tensor prefill_logits_out;

        if (use_event_sync) {
            Tensor logits_out;
            executor_->forward_logits(state, logits_out, pf_stream);
            Tensor last_logits = logits_out.slice(0, 1);
            int64_t vocab_shape[1] = {last_logits.shape[1]};
            last_logits = last_logits.reshape(1, vocab_shape);

            // Ban special tokens (e.g. Gemma-4 <|channel>) before greedy
            // argmax — otherwise the natural-argmax channel marker triggers
            // is_stop_token and the request finishes with 0 completion
            // tokens. Same logic as GraphExecutor::forward (executor.cu:88)
            // and apply_pre_sample (executor.cu) but inline here because
            // sample_greedy_device runs on raw logits without going through
            // either of those wrappers.
            if (state.banned_tokens != nullptr && state.n_banned_tokens > 0) {
                float* lp = static_cast<float*>(last_logits.data);
                int vocab = static_cast<int>(last_logits.shape[0]);
                float neg_inf = -1e30f;
                for (int bi = 0; bi < state.n_banned_tokens; bi++) {
                    int32_t tid = state.banned_tokens[bi];
                    if (tid >= 0 && tid < vocab) {
                        IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(lp + tid, &neg_inf, sizeof(float),
                                                           cudaMemcpyHostToDevice, pf_stream));
                    }
                }
            }

            sample_greedy_device(last_logits, executor_->d_sample_result(), h_sample_pinned_, pf_stream);

            if (!prefill_done_)
                (void)prefill_done_.create();
            cudaEventRecord(prefill_done_, pf_stream);

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
            }

            cudaEventSynchronize(prefill_done_);
            next_token = *h_sample_pinned_;
        } else if (req->logprobs) {
            executor_->forward_logits(state, prefill_logits_out, pf_stream);
            auto sampled = executor_->sample_from_logits(prefill_logits_out, state, pf_stream);
            next_token = sampled[0];

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
            }
        } else {
            next_token = executor_->forward(state, pf_stream);

            if (!pf_pool_used) {
                free_prefill_buffers(d_token_ids, d_positions, d_block_tables, d_context_lens, pf_stream);
            }
        }

        if (req->mirostat == 2)
            req->mirostat_mu = state.mirostat_mu;

        // Extract logprobs
        if (req->logprobs && prefill_logits_out.data != nullptr) {
            int vocab_size = static_cast<int>(prefill_logits_out.shape[prefill_logits_out.ndim - 1]);
            executor_->ensure_logits_pinned(vocab_size);

            const float* d_logits = static_cast<const float*>(prefill_logits_out.data);
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(executor_->h_logits_pinned(), d_logits,
                                               vocab_size * sizeof(float), cudaMemcpyDeviceToHost,
                                               pf_stream));
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(pf_stream));

            req->output_logprobs.push_back(build_logprob_info(executor_->h_logits_pinned(), vocab_size,
                                                              next_token, req->top_logprobs,
                                                              model_->tokenizer()));
        }

        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);

        Tokenizer* tok = model_->tokenizer();
        IMP_LOG_DEBUG("Prefill -> token %d (ctx=%d): id=%d [%s]", (int)req->output_tokens.size(),
                      req->context_len(), next_token, tok->decode_token(next_token).c_str());

        // MTP prefill: populate MTP KV cache with all prompt position hidden
        // states so the head enters decode with the same context as the main
        // model. Only the LAST chunk (where we have the complete tail of the
        // prompt in executor's hidden_ buffer) and only the non-chunked path
        // for now (chunked prefill would need per-chunk capture). Cost: ~n
        // extra MTP forwards, one-time per session.
        if (mtp_spec_decode_enabled() && offset == 0 && req->prefill_offset == 0) {
            // executor's hidden_ buffer holds [chunk_len, d_model] FP16 right
            // after forward_logits; that matches the whole prompt when
            // offset==0 and is_last_chunk==true.
            const int n_prompt = chunk_len;
            imp::Tensor h_view = executor_->view_hidden(n_prompt);
            if (h_view.data != nullptr) {
                mtp_prefill_prompt(req->input_tokens.data(), h_view.data, n_prompt);
            }
        }

        // Update constraint FSM
        constraints_.update(next_token);

        if (should_stop(*req, next_token) || static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
        } else {
            req->status = RequestStatus::DECODING;
            if (kv_manager_->prefix_caching_enabled()) {
                kv_manager_->register_block_hashes(req->id, req->input_tokens);
            }
        }
    }
}

// =====================================================================
// step_decode — process all decode requests (batched)
// =====================================================================

void Engine::step_decode(cudaStream_t dec_stream) {
    auto& decode_batch = sched_decode_batch_;
    const int kv_bs = kv_cache_raw_ ? kv_cache_raw_->block_size() : kKVBlockSize;

    // SSM/GDN: limit decode batch to 1 sequence
    if ((ssm_state_ || gdn_state_) && decode_batch.size() > 1) {
        decode_batch.resize(1);
    }

    // Allocate new KV blocks where needed
    valid_decode_.clear();
    auto& valid_decode = valid_decode_;

    for (auto& req : decode_batch) {
        int ctx_len = req->context_len();
        int blocks_needed = (ctx_len + kv_bs - 1) / kv_bs;
        const auto& block_table = kv_manager_->block_table(req->id);
        int blocks_have = static_cast<int>(block_table.size());

        if (blocks_needed > blocks_have) {
            int new_block = kv_manager_->append_block(req->id);
            if (new_block < 0) {
                int evicted = kv_manager_->evict_lru();
                if (evicted >= 0) {
                    new_block = kv_manager_->append_block(req->id);
                }
                if (new_block < 0) {
                    kv_manager_->free_sequence(req->id);
                    req->status = RequestStatus::CANCELLED;
                    continue;
                }
            }
        }

        // StreamingLLM smart KV cache: once context exceeds the threshold,
        // free middle blocks while keeping sinks + window. The decode kernel
        // skips the freed (-1 sentinel) slots via its own n_sinks logic.
        if (config_.streaming_kv_enabled) {
            int n_sinks = (config_.streaming_kv_n_sinks > 0) ? config_.streaming_kv_n_sinks : 4;
            int win = (config_.streaming_kv_window > 0) ? config_.streaming_kv_window
                                                        : model_->config().sliding_window;
            if (n_sinks > 0 && win > 0) {
                int threshold = (config_.streaming_kv_threshold > 0) ? config_.streaming_kv_threshold
                                                                     : (n_sinks + win + 2 * kv_bs);
                if (req->context_len() > threshold) {
                    kv_manager_->evict_middle_blocks(req->id, n_sinks, win);
                }
            }
        }
        valid_decode.push_back(req);
    }

    if (!valid_decode.empty()) {
        step_decode_forward(valid_decode, dec_stream);
    }
}

// =====================================================================
// step_decode_forward — build batch, run forward pass, sample, process
// =====================================================================

// Populate the InferenceState for a decode step from the uploaded GPU batch.
// Handles per-seq residual metadata (single-seq fast path vs multi-seq
// per-batch upload), sampling params, decode-step seed, penalties, recurrent
// state, and JSON/schema constrainer attach. Returns needs_logprobs so the
// caller knows whether to capture decode_logits_out for the logprobs pass.
void Engine::decode_build_inference_state_(GPUBatch& gpu_batch,
                                           std::vector<std::shared_ptr<Request>>& valid_decode,
                                           int max_ctx, cudaStream_t dec_stream,
                                           InferenceState& state, bool& needs_logprobs,
                                           bool& needs_json_mode, bool& needs_schema_mode) {
    state.token_ids = gpu_batch.d_token_ids;
    state.positions = gpu_batch.d_positions;
    state.n_tokens = gpu_batch.total_tokens;
    state.n_sequences = gpu_batch.n_sequences;
    state.max_blocks_per_seq = gpu_batch.max_blocks_per_seq;
    state.kv_cache = kv_cache_raw_;
    state.block_tables = gpu_batch.d_block_tables;
    state.context_lens = gpu_batch.d_context_lens;
    state.max_context_len = max_ctx;
    state.is_prefill = false;
    state.kv_manager = kv_manager_.get();
    if (kv_manager_ && kv_manager_->residual_enabled()) {
        // Allocate / refresh per-seq residual metadata for this decode step.
        // Slot allocation is idempotent (returns existing slot on re-call).
        const int N = gpu_batch.n_sequences;
        residual_meta_h_seq_ids_.resize(N);
        for (int i = 0; i < N; i++) {
            int sid = valid_decode[i]->id;
            residual_meta_h_seq_ids_[i] = sid;
            kv_manager_->allocate_residual_slot(sid);
        }
        if (N == 1) {
            // Single-seq path: kernel reads ring state from kv_manager's
            // persistent device buffers. Slot is a constant per-request
            // value uploaded into d_kv_slot_buf_[0]; only re-uploads when
            // it changes (i.e. when the active request rotates).
            state.kv_seq_id = valid_decode[0]->id;
            state.h_residual_seq_ids = residual_meta_h_seq_ids_.data();
            int slot_for_req = kv_manager_->residual_slot_of(valid_decode[0]->id);
            if (d_kv_slot_buf_ != nullptr) {
                if (d_kv_slot_last_uploaded_.empty() ||
                    d_kv_slot_last_uploaded_[0] != slot_for_req) {
                    cudaMemcpyAsync(d_kv_slot_buf_, &slot_for_req, sizeof(int),
                                    cudaMemcpyHostToDevice, dec_stream);
                    if (d_kv_slot_last_uploaded_.empty()) d_kv_slot_last_uploaded_.assign(1, -1);
                    d_kv_slot_last_uploaded_[0] = slot_for_req;
                }
                state.d_residual_seq_slots = d_kv_slot_buf_;
            }
        } else {
            // Multi-seq path: build per-batch metadata arrays + upload to
            // a per-step device buffer.
            residual_meta_h_slots_.resize(N);
            residual_meta_h_counts_.resize(N);
            residual_meta_h_widxes_.resize(N);
            for (int i = 0; i < N; i++) {
                int sid = residual_meta_h_seq_ids_[i];
                residual_meta_h_slots_[i] = kv_manager_->residual_slot_of(sid);
                auto rs = kv_manager_->residual_state(sid);
                residual_meta_h_counts_[i] = rs.fill_count;
                residual_meta_h_widxes_[i] = rs.write_idx;
            }
            const size_t meta_bytes = static_cast<size_t>(3) * N * sizeof(int);
            if (cudaMallocAsync(&residual_meta_d_buf_, meta_bytes, dec_stream) == cudaSuccess) {
                int* base = residual_meta_d_buf_;
                cudaMemcpyAsync(base + 0 * N, residual_meta_h_slots_.data(), N * sizeof(int),
                                cudaMemcpyHostToDevice, dec_stream);
                cudaMemcpyAsync(base + 1 * N, residual_meta_h_counts_.data(), N * sizeof(int),
                                cudaMemcpyHostToDevice, dec_stream);
                cudaMemcpyAsync(base + 2 * N, residual_meta_h_widxes_.data(), N * sizeof(int),
                                cudaMemcpyHostToDevice, dec_stream);
                state.d_residual_seq_slots = base + 0 * N;
                state.d_residual_counts = base + 1 * N;
                state.d_residual_write_idxes = base + 2 * N;
                state.h_residual_seq_ids = residual_meta_h_seq_ids_.data();
            }
        }
    }
    fill_sampling_params(*valid_decode[0], state);

    // Derive per-step seed: mix request seed with output count so each
    // decode step gets a different random draw.  Without this, seed=-1
    // falls back to a fixed 42 on every step, producing identical RNG
    // values and causing repetition loops on long structured outputs.
    state.seed = compute_step_seed(*valid_decode[0]);

    // Penalties (single-sequence only)
    if (gpu_batch.n_sequences == 1) {
        upload_penalties(*valid_decode[0], state, dec_stream);
    }

    // Recurrent state
    fill_recurrent_state(*valid_decode[0], state, false, dec_stream);

    // Check if any request needs logprobs or constrained mode
    needs_logprobs = false;
    needs_json_mode = false;
    needs_schema_mode = false;
    for (const auto& r : valid_decode) {
        if (r->logprobs)
            needs_logprobs = true;
        if (r->json_mode && r->json_schema.empty())
            needs_json_mode = true;
        if (!r->json_schema.empty())
            needs_schema_mode = true;
    }

    // Schema/JSON constraints for decode (reuse state from prefill)
    if (needs_schema_mode && valid_decode.size() == 1 && !valid_decode[0]->json_schema.empty()) {
        if (constraints_.has_schema()) {
            state.schema_constrainer = constraints_.schema_constrainer();
        }
    }
    if (needs_json_mode && valid_decode.size() == 1 && valid_decode[0]->json_mode) {
        // Lazily init if needed (decode might be first step with json_mode)
        if (!constraints_.has_json() && !constraints_.has_schema()) {
            constraints_.prepare(true, "", model_->tokenizer());
        }
        state.json_constrainer = constraints_.json_constrainer();
    }
}

void Engine::step_decode_forward(std::vector<std::shared_ptr<Request>>& valid_decode,
                                 cudaStream_t dec_stream) {
    // Switch workspace for decode
    if (executor_->has_decode_workspace() && valid_decode.size() == 1) {
        executor_->use_workspace(1);
    } else {
        if (executor_->active_workspace() == 1)
            executor_->use_workspace(0);
        (void)executor_->resize_workspace(static_cast<int>(valid_decode.size()), dec_stream);
    }

    // Build batched decode
    decode_builder_.reset();

    int max_ctx = 0;
    for (auto& req : valid_decode) {
        int ctx_len = req->context_len();
        max_ctx = std::max(max_ctx, ctx_len);

        int32_t last_token = req->output_tokens.empty() ? req->input_tokens.back()
                                                        : req->output_tokens.back();
        int position = ctx_len - 1;

        const auto& bt = kv_manager_->block_table(req->id);
        decode_builder_.add_decode_sequence(last_token, position, bt.data(), static_cast<int>(bt.size()),
                                            ctx_len);
    }

    Batch batch = decode_builder_.build();

    // Upload to GPU using pre-allocated pool
    GPUBatch gpu_batch;
    if (decode_batch_pool_.is_allocated()) {
        int pool_max = decode_batch_pool_.max_blocks_per_seq();
        if (batch.max_blocks_per_seq < pool_max) {
            int n_seq = batch.n_sequences;
            int old_stride = batch.max_blocks_per_seq;
            size_t needed = static_cast<size_t>(n_seq) * pool_max;
            padded_block_table_.resize(needed);
            std::memset(padded_block_table_.data(), 0, needed * sizeof(int));
            for (int s = 0; s < n_seq; s++) {
                for (int b = 0; b < old_stride; b++) {
                    padded_block_table_[s * pool_max + b] = batch.block_tables[s * old_stride + b];
                }
            }
            batch.block_tables.swap(padded_block_table_);
            batch.max_blocks_per_seq = pool_max;
        }
        gpu_batch = decode_batch_pool_.upload_into_pool(batch, dec_stream);
    } else {
        gpu_batch.upload(batch, dec_stream);
    }

    InferenceState state;
    bool needs_logprobs = false;
    bool needs_json_mode = false;
    bool needs_schema_mode = false;
    decode_build_inference_state_(gpu_batch, valid_decode, max_ctx, dec_stream, state, needs_logprobs,
                                  needs_json_mode, needs_schema_mode);

    // Per-request sampling lambda
    auto sample_per_request = [&](const Tensor& logits) -> std::vector<int32_t> {
        int n = static_cast<int>(valid_decode.size());

        if (n == 1) {
            auto& req = valid_decode[0];
            int32_t tok = executor_->sample_single_from_logits(logits, state, dec_stream);
            if (state.mirostat == 2)
                req->mirostat_mu = state.mirostat_mu;
            return {tok};
        }

        std::vector<int32_t> result(n);
        for (int i = 0; i < n; i++) {
            auto& req = valid_decode[i];
            InferenceState per_state = state;
            fill_sampling_params(*req, per_state);
            // Per-step seed (same fix as single-sequence path)
            per_state.seed = compute_step_seed(*req);
            per_state.penalty_tokens = nullptr;
            per_state.n_penalty_tokens = 0;
            bool req_needs_pen = (req->repetition_penalty != 1.0f || req->frequency_penalty != 0.0f ||
                                  req->presence_penalty != 0.0f);
            if (req_needs_pen && !req->output_tokens.empty() && d_penalty_tokens_) {
                size_t rn = req->output_tokens.size();
                if (rn <= d_penalty_tokens_capacity_) {
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_penalty_tokens_, req->output_tokens.data(),
                                                       rn * sizeof(int32_t), cudaMemcpyHostToDevice,
                                                       dec_stream));
                    per_state.penalty_tokens = d_penalty_tokens_;
                    per_state.n_penalty_tokens = static_cast<int>(rn);
                }
            }
            per_state.n_sequences = 1;
            Tensor seq_logits = logits.slice(i, i + 1);
            result[i] = executor_->sample_single_from_logits(seq_logits, per_state, dec_stream);
            if (per_state.mirostat == 2)
                req->mirostat_mu = per_state.mirostat_mu;
        }
        return result;
    };

    // Execute forward pass (piecewise CUDA Graph: forward in graph,
    // sampling always eager — per-batch-size graph pool avoids
    // re-capture when continuous batching changes batch size)
    std::vector<int32_t> tokens;
    Tensor decode_logits_out;

    const bool profiling = RuntimeConfig::current().diagnostics.profile;
    int graph_idx = gpu_batch.n_sequences - 1;
    if (config_.use_cuda_graphs && !profiling && gpu_batch.n_sequences > 0 && graph_idx < kMaxGraphPoolSize &&
        decode_batch_pool_.is_allocated()) {
        auto& graph_runner = decode_graph_pool_[graph_idx];

        // P5 §2.2 M4: pow-2 bucket the max_blocks_per_seq before comparing.
        // The decode_batch_pool_ padding (engine.cpp:2456-2468) already lifts
        // batch.max_blocks_per_seq to pool_max, so in practice this comparison
        // never trips during steady-state decode — but if a future path
        // bypasses the pool (e.g. spec-decode verify), pow-2 bucketing caps
        // the re-capture frequency to log2(max_blocks) events per decode
        // instead of one per 16-token boundary. Re-capture cost is
        // dominated by the eager forward that builds the new graph
        // (~5-10 ms on Qwen3-8B Q8_0) so the bucket pays back across long
        // contexts.
        auto bucket_pow2 = [](int x) -> int {
            if (x <= 1) return 1;
            int b = 1;
            while (b < x) b <<= 1;
            return b;
        };
        const int bucketed_max_blocks = bucket_pow2(gpu_batch.max_blocks_per_seq);
        if (bucketed_max_blocks != last_decode_max_blocks_per_graph_[graph_idx]) {
            // Topology stable across max_blocks growth (same kernels, only
            // grid dims / params differ) — cudaGraphExecUpdate handles this
            // without tearing down the exec + graph mem pool.
            graph_runner.invalidate_for_update();
            last_decode_max_blocks_per_graph_[graph_idx] = bucketed_max_blocks;
        }
        // Graph captures ONLY forward_logits — sampling runs eager after
        Tensor logits_out;
        graph_runner.set_decode_fn(
            [this, &state, &logits_out](cudaStream_t s) { executor_->forward_logits(state, logits_out, s); });
        graph_runner.execute(dec_stream);

        if (logits_out.data == nullptr) {
            logits_out = executor_->get_logits_view(gpu_batch.n_sequences);
        }
        // Eager sampling (handles all modes: greedy, top-k/p, penalties,
        // force_token, constraints, logprobs, mirostat)
        tokens = sample_per_request(logits_out);
        if (needs_logprobs)
            decode_logits_out = logits_out;
    } else {
        executor_->forward_logits(state, decode_logits_out, dec_stream);
        tokens = sample_per_request(decode_logits_out);
    }

    if (!decode_batch_pool_.is_allocated()) {
        gpu_batch.free();
    }

    // Free per-step residual metadata buffer (allocated in step_decode_forward
    // when residual is enabled). cudaFreeAsync orders behind the just-issued
    // forward + sample on dec_stream.
    if (residual_meta_d_buf_ != nullptr) {
        IMP_CUDA_CHECK_LOG(cudaFreeAsync(residual_meta_d_buf_, dec_stream));
        residual_meta_d_buf_ = nullptr;
    }

    // Phase 3.5 telemetry: measure MTP-draft prediction accuracy without
    // changing generation. Single-sequence only (batch=1 simplifies hidden-
    // state addressing). Skipped when MTP is disabled or the workspace was
    // allocated without attention dims (older callers).
    if (mtp_spec_decode_enabled() && model_ && model_->mtp_.has_value() &&
        model_->mtp_->loaded && gpu_batch.n_sequences == 1 && !tokens.empty()) {
        const int32_t next_token = tokens[0];
        if (mtp_pending_prediction_ >= 0) {
            mtp_accuracy_.total++;
            bool match = (mtp_pending_prediction_ == next_token);
            if (match) mtp_accuracy_.matches++;
            // Optional verbose log: prints (predicted, actual, match) with
            // decoded strings so accept patterns can be analyzed offline.
            const bool s_pattern_log = RuntimeConfig::current().diagnostics.mtp_pattern_log;
            if (s_pattern_log) {
                Tokenizer* tok = model_->tokenizer();
                std::string ps = tok ? tok->decode_token(mtp_pending_prediction_) : std::string();
                std::string as = tok ? tok->decode_token(next_token) : std::string();
                IMP_LOG_INFO("MTP-PAT %s pred=%d '%s' actual=%d '%s'",
                             match ? "+" : "-",
                             mtp_pending_prediction_, ps.c_str(),
                             next_token, as.c_str());
            }
        }
        // K-chain measurement: verify pending predictions, then draft fresh.
        // Current position is the index of the token we just emitted (=
        // length of input + already-emitted output minus 1).
        const int cur_pos = valid_decode[0]->context_len() - 1;
        // Verify any pending predictions whose intended position is cur_pos.
        if (!mtp_pending_chain_.empty()) {
            // Drop stale (intended < cur_pos) — shouldn't happen in K=1 path
            // but possible if the engine skips/restarts mid-chain.
            mtp_pending_chain_.erase(
                std::remove_if(mtp_pending_chain_.begin(), mtp_pending_chain_.end(),
                               [cur_pos](const MtpChainEntry& e) {
                                   return e.intended_position < cur_pos;
                               }),
                mtp_pending_chain_.end());
            for (auto it = mtp_pending_chain_.begin(); it != mtp_pending_chain_.end();) {
                if (it->intended_position == cur_pos) {
                    if (static_cast<int>(mtp_chain_accept_.size()) <= it->lookahead) {
                        mtp_chain_accept_.resize(it->lookahead + 1);
                    }
                    mtp_chain_accept_[it->lookahead].total++;
                    if (it->prediction == next_token) {
                        mtp_chain_accept_[it->lookahead].matches++;
                    }
                    it = mtp_pending_chain_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        Tensor h_view = executor_->view_hidden(1);  // [1, d_model] FP16
        if (h_view.data != nullptr) {
            const int hidden_dim = model_->config_.d_model;
            const int vocab_size = model_->config_.vocab_size;

            // Optional: apply the main model's output norm before passing
            // h_prev to MTP. Upstream vllm passes post-RMSNorm hidden states
            // in some MTP variants; gate by env so we can A/B.
            const bool s_pre_norm_h = RuntimeConfig::current().diagnostics.mtp_prenorm_h;
            const void* h_for_mtp = h_view.data;
            // Scratch buffer for the normalized variant (allocated once).
            static void* s_h_normed = nullptr;
            if (s_pre_norm_h) {
                if (s_h_normed == nullptr) {
                    cudaMalloc(&s_h_normed, hidden_dim * sizeof(__half));
                }
                int64_t hd_shape[2] = {1, hidden_dim};
                Tensor in_view (h_view.data, QType::F16, 2, hd_shape, true);
                Tensor out_view(s_h_normed,  QType::F16, 2, hd_shape, true);
                imp::rmsnorm(in_view, model_->output_norm(), out_view,
                             model_->config_.rms_norm_eps, decode_stream(),
                             model_->config_.norm_weight_offset);
                h_for_mtp = s_h_normed;
            }

            // K-chain draft. K=mtp_spec_k_. For each step k=0..K-1:
            //   - input: (prev_token_k, h_prev_k)
            //   - output: prediction_k, ws.d_h_final updated for next iter
            //   - chain: prev_token_{k+1} = prediction_k, h_prev_{k+1} = d_h_final
            //
            // Cache roll-back: each chained call appends to MTP KV cache. Only
            // the first chain step's append should persist (it represents what
            // the main model actually does next). Roll back to mtp_pos_saved
            // after K-1 speculative steps so the real cache stays aligned.
            auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
            const int K = std::max(1, mtp_spec_k_);
            const int mtp_pos_before = ws->mtp_pos;
            int chain_prev_tok = next_token;
            const void* chain_h_prev = h_for_mtp;
            for (int k = 0; k < K; ++k) {
                int prediction = -1;
                if (!mtp_draft_one(chain_prev_tok, chain_h_prev, hidden_dim, vocab_size,
                                    &prediction)) {
                    break;
                }
                mtp_pending_chain_.push_back({prediction, k, cur_pos + 1 + k});
                // For k=0 only, also feed pending_prediction_ (legacy 1-step
                // accuracy counter remains in sync with chain_accept_[0]).
                if (k == 0) mtp_pending_prediction_ = prediction;
                // Chain: next iter uses this prediction + the MTP's own h_final.
                chain_prev_tok = prediction;
                chain_h_prev   = ws->d_h_final;
            }
            // Roll back the speculative cache writes from K-1 chained steps.
            // The first chained step (k=0) IS the real "next step" prediction
            // and matches what would have been drafted in K=1 mode — keep it.
            ws->mtp_pos = std::min(ws->mtp_pos, mtp_pos_before + 1);
        } else {
            mtp_pending_prediction_ = -1;
        }
    } else {
        mtp_pending_prediction_ = -1;  // batch>1 or MTP off → clear pending
    }

    // Process outputs: logprobs extraction + token distribution
    step_decode_process_outputs(valid_decode, tokens, decode_logits_out, needs_logprobs, needs_json_mode,
                                needs_schema_mode, dec_stream);
}

// =====================================================================
// step_decode_process_outputs — extract logprobs, distribute tokens,
//                                try async graph loop
// =====================================================================

void Engine::step_decode_process_outputs(std::vector<std::shared_ptr<Request>>& valid_decode,
                                         const std::vector<int32_t>& tokens, const Tensor& decode_logits_out,
                                         bool needs_logprobs, bool needs_json_mode, bool needs_schema_mode,
                                         cudaStream_t dec_stream) {
    Tokenizer* tok = model_->tokenizer();

    // Extract logprobs
    if (needs_logprobs && decode_logits_out.data != nullptr) {
        int vocab_size = static_cast<int>(decode_logits_out.shape[decode_logits_out.ndim - 1]);
        int n_lp = 0;
        for (const auto& r : valid_decode)
            if (r->logprobs)
                n_lp++;
        executor_->ensure_logits_pinned(vocab_size * n_lp);
        float* h_base = executor_->h_logits_pinned();

        int slot = 0;
        for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
            if (!valid_decode[i]->logprobs)
                continue;
            const float* d_logits = static_cast<const float*>(decode_logits_out.data) +
                                    static_cast<size_t>(i) * vocab_size;
            IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(h_base + static_cast<size_t>(slot) * vocab_size, d_logits,
                                               vocab_size * sizeof(float), cudaMemcpyDeviceToHost,
                                               dec_stream));
            slot++;
        }
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(dec_stream));

        slot = 0;
        for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
            auto& req = valid_decode[i];
            if (!req->logprobs)
                continue;

            float* h_logits = h_base + static_cast<size_t>(slot) * vocab_size;
            req->output_logprobs.push_back(
                build_logprob_info(h_logits, vocab_size, tokens[i], req->top_logprobs, tok));
            slot++;
        }
    }

    // Distribute sampled tokens back to requests
    for (int i = 0; i < static_cast<int>(valid_decode.size()); i++) {
        auto& req = valid_decode[i];
        int32_t next_token = tokens[i];

        req->output_tokens.push_back(next_token);
        track_think_state(*req, next_token);

        IMP_LOG_DEBUG("Decode step %d (ctx=%d, pos=%d): id=%d [%s]", (int)req->output_tokens.size(),
                      req->context_len(), req->context_len() - 1, next_token,
                      tok->decode_token(next_token).c_str());

        if (should_stop(*req, next_token) || static_cast<int>(req->output_tokens.size()) >= req->max_tokens) {
            finish_request(req);
        }

        constraints_.update(next_token);
        kv_manager_->touch(req->id);
    }

    // Try async graph loop after first decode step.
    // Think budget is now handled device-side in post_decode_step_kernel.
    if (decode_graph_pool_[0].is_ready() && valid_decode.size() == 1 && !offload_mgr_ &&
        config_.use_cuda_graphs && !async_graph_runner_.is_setup() && !needs_logprobs && !needs_json_mode &&
        !needs_schema_mode) {
        auto& dreq = valid_decode[0];
        // forward_decode_async only implements banned_tokens + rep/freq/presence
        // penalties device-side. Any sampling feature that requires host-side
        // logic (logit_bias, mirostat, typical_p, min_p, DRY) would be silently
        // skipped inside the captured graph — stay on the eager path instead.
        const bool async_compatible = dreq->logit_bias.empty() && dreq->mirostat == 0 &&
                                      dreq->dry_multiplier == 0.0f && dreq->min_p == 0.0f &&
                                      dreq->typical_p >= 1.0f &&
                                      // Phase 3.5 MTP telemetry hooks the per-step path; the async
                                      // conditional-graph loop bypasses it. Stay eager when MTP is on
                                      // so accuracy measurement covers the whole generation.
                                      !mtp_spec_decode_enabled();
        // Text-fallback think tracking: when <think>/</think> are not single
        // control-token IDs (NVFP4 SafeTensors for Qwen3 / Qwen3.5 / Qwen3.6
        // ship them as added_tokens with special=False, leaving think_end_id_
        // at -1), the graph kernel's device-side `in_think` predicate is
        // permanently false. eos_id then terminates the loop the moment the
        // model emits <|endoftext|> after an empty </think>, returning a
        // 3-token completion ("<", "answer", ">") with the actual answer
        // never sampled. Host-side track_think_state runs literal-string
        // matching per token but only fires in the eager step_decode path,
        // so route through it whenever the request started inside a think
        // block on a model that lacks single-token think markers.
        const bool needs_eager_for_text_think = dreq->in_think_block && think_end_id_ < 0;
        if (async_compatible && !needs_eager_for_text_think && dreq->status == RequestStatus::DECODING &&
            !dreq->output_tokens.empty()) {
            int32_t last_token = dreq->output_tokens.back();
            try_launch_async_graph_loop(dreq, last_token, dec_stream);
        }
    }
}
}  // namespace imp
