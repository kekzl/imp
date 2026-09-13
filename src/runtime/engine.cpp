#include "runtime/engine.h"
#include "core/buffer.h"
#include "vision/image_processor.h"
#include "vision/qwen3vl_vision_load.h"
#include "vision/vision_pipeline.h"
#include "runtime/request.h"
#include "lora/lora_adapter.h"
#include "runtime/engine_internal.h"
#include "runtime/think_stop_logic.h"
#include "runtime/config.h"
#include "core/process_diag.h"
#include "runtime/process_diag_install.h"
#include "runtime/vram_budget.h"
#include "runtime/batch.h"
#include "compute/mtp_forward.h"
#include "compute/rope.h"  // rope_yarn_corr_dims (MTP rope-scaling parity)
#include "compute/encoder_forward.h"
#include "memory/kv_cache.h"
#include "memory/mem_account.h"
#include "memory/engine_arena.h"
#include "memory/graph_slots.h"
#include "memory/plan.h"
#include "exec/workspace_sizes.h"
#include "memory/backend.h"
#include "memory/vram_query.h"
#include "model/gguf_loader.h"
#include "model/chat_template.h"
#include "compute/ffn_sparsity_probe.h"
#include "compute/gemm.h"
#include "compute/mmq_q8_imma.h"
#include "compute/gemm_capture_fp16_sm120.h"
#include "compute/gemm_cutlass_grouped_3x.h"
#include "compute/attention_cublas.h"
#include "compute/gemm_grouped.h"
#include "compute/sampling.h"
#include "compute/attention.h"
#include "compute/layernorm.h"
#include "core/logging.h"
#include "core/cuda_static_reset.h"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <functional>
#include <vector>

namespace imp {

// File-local helpers live in runtime/engine_internal.h, shared by the
// per-subsystem engine_*.cpp translation units.

Engine::~Engine() {
    // Cross-model CUDA-error-leak guard: the CUDA error state is per primary
    // context, NOT per Engine, so a sticky error from this model's workload
    // survives imp_context_free/imp_model_free and corrupts the NEXT model's
    // first cudaGetLastError()-guarded kernel (e.g. gemm_cutlass_sm120.cu). Drain it here.
    if (cudaError_t leaked = cudaGetLastError(); leaked != cudaSuccess) {
        IMP_LOG_WARN(
            "Engine teardown: cleared a leaked CUDA error (%s) so it cannot "
            "corrupt the next model loaded in this process",
            cudaGetErrorString(leaked));
    }

    // Phase-0 VRAM audit: stop the peak sampler and emit the final table.
    // Teardown allocates nothing but frees plenty; leaving the process in
    // Serving would make the next engine's init look like an I2 violation.
    set_alloc_phase(AllocPhase::Loading);
    free_batch_verify_bufs_();
    if (const uint64_t n = steady_state_allocations(); n > 0) {
        IMP_LOG_WARN(
            "I2: %llu device allocation(s) were made while serving — "
            "see the per-tag warnings above (criterion 3 requires zero)",
            static_cast<unsigned long long>(n));
    }
    MemAccount::instance().sampler_stop();
    MemAccount::instance().report("shutdown");
    // FFN sparsity probe: drain per-layer counters to stderr if any decode
    // steps ran with it enabled. Counters are an arena slice, so this must run
    // BEFORE the arena closes.
    flush_ffn_sparsity_probe_log();
    engine_arena_close();
    // The arena is gone; every module static that took a slice from it now
    // dangles. reset_static_cuda_state() re-arms them (previously wired only to
    // imp_api_suspend.cpp, so a second engine reused freed memory via
    // gemm_init()'s `if (!s_workspace)` guard). Must run AFTER the arena closes.
    reset_static_cuda_state();

    // Save prefix cache to disk before shutdown (dense only: hybrid reuse
    // needs recurrent snapshots, which are device-resident and not persisted)
    if (kv_manager_ && !config_.prefix_cache_path.empty() && kv_manager_->prefix_caching_enabled() &&
        !ssm_state_) {
        kv_manager_->save_prefix_cache(config_.prefix_cache_path, model_fingerprint_(), stream_);
    }

    gemm_cleanup();
    sampling_cleanup();
    if (async_graph_runner_.is_setup()) {
        async_graph_runner_.cleanup();
    }
    async_graph_runner_.drop_spare();
    // Strictly after the runner teardown above: closing the pool frees the
    // slots, so an outstanding lease here means something still holds
    // addresses into memory about to go away. The pool logs that as an error.
    graph_slot_pool().close();
    release_async_block_tables_();
    // The constrained pipeline's pinned landing and event outlive one
    // pipeline (teardown keeps them); this is their one release.
    cpipe_.h_token.reset();
    if (cpipe_.ev) {
        IMP_CUDA_CHECK_LOG(cudaEventDestroy(cpipe_.ev));
        cpipe_.ev = nullptr;
    }
    if (swa_snap_slab_) {
        IMP_CUDA_CHECK_LOG(cudaFree(swa_snap_slab_));
        swa_snap_slab_ = nullptr;
    }
    free_mrope_buffers_();
    if (d_penalty_tokens_) {
        vram_alloc_.free(d_penalty_tokens_);
        d_penalty_tokens_ = nullptr;
    }
    if (d_embed_pool_scratch_) {
        vram_alloc_.free(d_embed_pool_scratch_);
        d_embed_pool_scratch_ = nullptr;
    }
    if (d_token_is_whitespace_) {
        vram_alloc_.free(d_token_is_whitespace_);
        d_token_is_whitespace_ = nullptr;
    }
    if (d_kv_slot_buf_) {
        cudaFree(d_kv_slot_buf_);
        d_kv_slot_buf_ = nullptr;
    }
    if (residual_meta_d_buf_) {
        cudaFree(residual_meta_d_buf_);
        residual_meta_d_buf_ = nullptr;
        residual_meta_capacity_ = 0;
    }
    abandon_decode_pipeline();
    for (int p = 0; p < 2; ++p) {
        // T5b owners release themselves; the d_* mirrors are views into them, so
        // they only need nulling (memory/host_pinned.h).
        h_bt_patch_off_[p].reset();
        h_bt_patch_val_[p].reset();
        h_hist_pos_[p].reset();
        d_bt_patch_off_[p] = nullptr;
        d_bt_patch_val_[p] = nullptr;
        d_hist_pos_[p] = nullptr;
    }
    if (d_pipe_hist_) {
        IMP_CUDA_CHECK_LOG(cudaFree(d_pipe_hist_));
        d_pipe_hist_ = nullptr;
    }
    log_spec_stats_();
    free_spec_buffers_();
    if (prefill_pool_) {
        vram_alloc_.free(prefill_pool_);
        prefill_pool_ = nullptr;
    }
    if (pf_staging_evt_) {
        IMP_CUDA_CHECK_LOG(cudaEventDestroy(pf_staging_evt_));
        pf_staging_evt_ = nullptr;
    }
    // Encoder embedder workspace cleanup (#836)
    if (encoder_ws_storage_) {
        auto* ews = static_cast<imp::EncoderWorkspace*>(encoder_ws_storage_);
        imp::encoder_workspace_free(*ews);
        delete ews;
        encoder_ws_storage_ = nullptr;
    }
    if (mtp_ws_storage_) {
        auto* ws = static_cast<imp::MtpDraftWorkspace*>(mtp_ws_storage_);
        imp::mtp_workspace_free(*ws);
        delete ws;
        mtp_ws_storage_ = nullptr;
        mtp_spec_k_ = 0;
    }
    // stream_, prefill_done_, decode_done_ cleaned up by CudaStream/CudaEvent RAII
    // vision_ cleaned up by VisionPipeline RAII
}


// (mtp_prefill_prompt -> per-chunk mtp_prefill_feed_chunk in engine_spec_mtp.cpp.)

bool Engine::encoder_embed(std::span<const int32_t> tokens, std::vector<float>& out) {
    if (encoder_ws_storage_ == nullptr || !model_) {
        IMP_LOG_ERROR("encoder_embed: no encoder workspace (not an encoder model?)");
        return false;
    }
    auto* ews = static_cast<imp::EncoderWorkspace*>(encoder_ws_storage_);
    out.resize(model_->config_.d_model);
    return imp::encoder_embed(*model_, *ews, tokens, out.data(), stream_);
}

cudaStream_t Engine::prefill_stream() const {
    return (config_.use_green_contexts && green_ctx_.is_available()) ? green_ctx_.prefill_stream() : stream_;
}

cudaStream_t Engine::decode_stream() const {
    return (config_.use_green_contexts && green_ctx_.is_available()) ? green_ctx_.decode_stream() : stream_;
}

void Engine::reset_ssm_state(int seq_id) {
    // Public teardown entry point (API re-prefill / context reset / server
    // cancellation): must return the slot to the free list or it leaks (these
    // paths bypass finish_request), exhausting the pool onto id%cap aliasing.
    auto it = recurrent_slot_of_.find(seq_id);
    if (ssm_state_) {
        const int cap = ssm_state_->max_sequences();
        int slot = (it != recurrent_slot_of_.end()) ? it->second : (cap > 0 ? seq_id % cap : 0);
        ssm_state_->reset_sequence(slot, stream_);
    }
    release_recurrent_slot_(seq_id);
    mtp_release_(seq_id);
}

void Engine::reset_batch_pool_cache() { decode_batch_pool_.reset_upload_cache(); }

void Engine::invalidate_graphs() {
    // Exception/reset path: an in-flight pipelined decode step must not be
    // waited on (the context may be poisoned), drop it and release the
    // deferred KV so the sequences don't leak blocks.
    abandon_decode_pipeline();

    // Preserve decode_graph_pool_ across context resets: decode-step topology
    // (forward_logits) doesn't change between requests, inputs upload fresh
    // each step via the batch pool. Re-capturing every reset costs ~100ms.
    //
    // The conditional graph runner MUST be invalidated: it captures the full
    // decode loop including token feedback, stop conditions, and request-specific
    // KV block pointers.
    if (async_graph_runner_.is_setup()) {
        async_graph_runner_.cleanup();
    }
    async_graph_runner_.drop_spare();
    async_graph_req_ = nullptr;
    async_pending_tokens_.clear();
    async_pending_cursor_ = 0;

    // The pipelined constrained decode holds a captured forward graph plus
    // request-specific device state, same invalidation requirement as the
    // conditional runner.
    if (cpipe_.active)
        teardown_constrained_pipeline(/*synchronize=*/true);

    // Captured verify-chunk graphs (#847) baked the forward as well (incl.
    // any active LoRA kernels/pointers), recapture costs two verify steps.
    free_spec_graphs_();
    // Verify-in-loop (#1055): an in-flight burst must complete before its
    // baked buffers/KV go away; the runner recaptures lazily afterwards.

    // #874 safety net: an exception unwound past an active prefill-chunk
    // capture leaves the stream in capture state, wedging the server
    // permanently. Close any stray capture and drop the prefill runner here.
    prefill_graph_runner_.invalidate();
    last_prefill_chunk_len_ = -1;
    last_prefill_block_count_ = -1;
    abort_stream_capture(prefill_stream());
}

int Engine::lora_load(const std::string& path) {
    auto a = std::make_unique<LoraAdapter>();
    if (!a->load(path, model_ ? model_->n_layers() : 0))
        return 0;
    // The adapter's declared widths become kernel extents (K is read from the
    // projection input, N written to its output), so they are held against
    // the base model here, not discovered on device (AUDIT_arch_2026 F1-6).
    if (model_) {
        const auto& cfg = model_->config();
        const int hd = cfg.head_dim > 0 ? cfg.head_dim : cfg.d_model / std::max(cfg.n_heads, 1);
        const LoraDims dims{cfg.d_model, cfg.n_heads * hd, cfg.n_kv_heads * hd, cfg.d_ff};
        std::string why;
        if (!a->check_dims(dims, &why)) {
            IMP_LOG_ERROR("LoRA: %s refused: %s", path.c_str(), why.c_str());
            return 0;
        }
    }
    lora_adapters_.push_back(std::move(a));
    return static_cast<int>(lora_adapters_.size());  // 1-based id
}

bool Engine::lora_set(int id) {
    if (id < 0 || id > static_cast<int>(lora_adapters_.size()))
        return false;
    if (id == active_lora_)
        return true;
    active_lora_ = id;
    executor_->set_lora(id == 0 ? nullptr : lora_adapters_[id - 1].get());
    // Decode graphs captured the previous forward (with/without the LoRA
    // kernels and with the old adapter's pointers), drop everything,
    // including the per-batch pool that invalidate_graphs() preserves.
    invalidate_graphs();
    for (auto& g : decode_graph_pool_)
        g.invalidate();
    IMP_LOG_INFO("LoRA: active adapter -> %d%s", id, id == 0 ? " (base)" : "");
    return true;
}

// Seed of the prefix-cache hash chain: the image it carries and the adapter it
// runs under. Two prompts with the same tokens but a different image or LoRA
// adapter must never share KV (AUDIT_arch_2026 E-1). 0 for plain text/base model.
// Computed once at admission; five call sites read it and must agree.
size_t Engine::prefix_salt_(const Request& req) const {
    size_t salt = req.vision_content_hash;
    if (active_lora_ != 0)
        salt ^= (static_cast<size_t>(active_lora_) + 1) * 0x9E3779B97F4A7C15ULL;
    return salt;
}

size_t Engine::effective_free_vram() const {
    // Budget-aware view (installed in init from config_.vram_budget_mb): uses
    // vram_query's process baseline delta, not global device usage, so a
    // co-tenant server's memory is never mis-charged to this process.
    size_t free_mem = 0;
    if (!vram_budget_mem_get_info(&free_mem, nullptr))
        return 0;
    return free_mem;
}

void Engine::finish_request(std::shared_ptr<Request>& req) {
    req->status = RequestStatus::FINISHED;
    finish_request_release_(req);
}

// KV/slot release half of finish_request, split out so pipelined batched
// decode can mark a row FINISHED while deferring the KV free until the
// in-flight chained step (which still writes this row's next slot) completes (bd_pipe_.deferred_release).
void Engine::finish_request_release_(std::shared_ptr<Request>& req) {
    // Publish under the salt the prefill looked these blocks up with (image
    // hash, adapter); an image request without a hash was refused the cache
    // at prefill and must not enter it here either.
    const bool had_image = req->n_vision_tokens > 0 || req->image || req->vision_emb;
    if (kv_manager_->prefix_caching_enabled() && (!had_image || req->vision_content_hash != 0)) {
        // Register input AND generated tokens, minus the final sampled token
        // (never forwarded, no KV entry; the spec-verify bonus token is the
        // same). Hashing the live reply KV turns the next agent turn's resend
        // into a prefix-cache hit instead of a full re-prefill.
        if (req->output_tokens.size() > 1) {
            std::vector<int32_t> forwarded;
            forwarded.reserve(req->input_tokens.size() + req->output_tokens.size() - 1);
            forwarded.insert(forwarded.end(), req->input_tokens.begin(), req->input_tokens.end());
            forwarded.insert(forwarded.end(), req->output_tokens.begin(), req->output_tokens.end() - 1);
            kv_manager_->register_block_hashes(req->id, forwarded, req->prefix_salt);
            // SWA window snapshot over the SAME span: without it the hashed
            // generated blocks are unusable under SWA sizing (reuse limit stops
            // at the last snapshot boundary). Must run before free_sequence
            // recycles the blocks; hard_sync because they're re-allocatable the moment it returns.
            maybe_save_swa_snapshot_span_(req->id, forwarded, stream_, /*hard_sync=*/true);
        } else {
            kv_manager_->register_block_hashes(req->id, req->input_tokens, req->prefix_salt);
        }
        // cache_control / cache_prompt: protect the prompt's full blocks from
        // eviction (must run before free_sequence, pinning needs the live block
        // table). A breakpoint boundary (#1046) caps the pin at the last cache_control marker.
        if (req->pin_kv_prefix) {
            int pin_tokens = static_cast<int>(req->input_tokens.size());
            if (req->pin_kv_prefix_tokens >= 0 && req->pin_kv_prefix_tokens < pin_tokens)
                pin_tokens = req->pin_kv_prefix_tokens;
            int full_blocks = pin_tokens / kv_manager_->kv_cache()->block_size();
            if (full_blocks > 0)
                kv_manager_->pin_prefix(req->id, full_blocks);
        }
    }
    kv_manager_->free_sequence(req->id);
    release_recurrent_slot_(req->id);
    mtp_release_(req->id);
    req->recurrent_restore.reset();  // release the snapshot buffer for recycling
    spec_suffix_idx_.erase(req->id);
    if (req->constraints)
        constraints_return_(std::move(req->constraints));
    // Server visibility: the engine outlives requests, so cumulative
    // speculation telemetry is logged per request end (no-op when idle).
    if (spec_any_drafter_enabled_(*req))
        log_spec_stats_();
}

void Engine::cancel_sequence_(const std::shared_ptr<Request>& req) {
    // Abnormal end of a request, counterpart to finish_request's graceful one -
    // both must release the same per-request resources, including the
    // recurrent-state slot: a fixed-size pool that, once empty, falls back to
    // `id % cap` aliasing and shares one SSM state across sequences (#1632).
    // Not shared with finish_request: prefix pinning (not worth it on cancel)
    // and speculation telemetry (no completed generation to report).
    kv_manager_->free_sequence(req->id);
    release_recurrent_slot_(req->id);
    mtp_release_(req->id);
    req->recurrent_restore.reset();
    req->swa_restore.reset();
    spec_suffix_idx_.erase(req->id);
    if (req->constraints)
        constraints_return_(std::move(req->constraints));
}

void Engine::score_capture_(Request& req, const Tensor& logits, cudaStream_t stream) {
    if (req.score_token_ids.empty() || logits.data == nullptr)
        return;
    // logits is [n_rows, vocab]; a cross-encoder's verdict lives at the LAST
    // position, which is the row the sampler would have used.
    const int64_t n_rows = logits.ndim >= 2 ? logits.shape[0] : 1;
    const int64_t vocab = logits.ndim >= 2 ? logits.shape[1] : logits.shape[0];
    const float* last_row = static_cast<const float*>(logits.data) +
                            static_cast<size_t>(n_rows - 1) * static_cast<size_t>(vocab);
    req.score_out.assign(req.score_token_ids.size(), 0.0f);
    // One D2H per id: the ids are a handful (yes/no), so a full-row copy would
    // move 600 KB to read 8 bytes.
    for (size_t i = 0; i < req.score_token_ids.size(); i++) {
        const int32_t id = req.score_token_ids[i];
        if (id < 0 || id >= static_cast<int32_t>(vocab))
            continue;
        IMP_CUDA_CHECK_LOG(
            cudaMemcpyAsync(&req.score_out[i], last_row + id, sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
}

void Engine::embed_accumulate_chunk_(Request& req, int chunk_len, cudaStream_t stream) {
    const int d_model = static_cast<int>(model_->config().d_model);
    if (!d_embed_pool_scratch_) {
        d_embed_pool_scratch_ = static_cast<float*>(
            vram_alloc_.allocate(static_cast<size_t>(d_model) * sizeof(float), "embed_pool_scratch"));
        if (!d_embed_pool_scratch_) {
            IMP_LOG_ERROR("VRAMAllocator failed for embed pool scratch (%d floats)", d_model);
            return;
        }
    }
    executor_->pool_hidden_sum(chunk_len, d_embed_pool_scratch_, stream);
    std::vector<float> partial(d_model);
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(partial.data(), d_embed_pool_scratch_,
                                       static_cast<size_t>(d_model) * sizeof(float), cudaMemcpyDeviceToHost,
                                       stream));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    if (req.embedding_out.empty())
        req.embedding_out.assign(d_model, 0.0f);
    for (int d = 0; d < d_model; d++)
        req.embedding_out[d] += partial[d];
}

void Engine::ensure_constraints_(const std::shared_ptr<Request>& req) {
    const bool tool_enforced = !req->tool_constraint_tools.empty();
    const bool regex_enforced = !req->regex_pattern.empty();
    const bool grammar_enforced = !req->grammar.empty();
    if (!(req->json_mode || !req->json_schema.empty() || tool_enforced || regex_enforced ||
          grammar_enforced) ||
        req->constraints)
        return;

    // Pool key: tool-enforced requests key by their tool-call signature so a
    // pooled manager with matching classified tables is reused (#1002). A
    // regex/grammar request keys by its source for the same reason.
    const std::string pool_key_regex = regex_enforced ? ("regex:" + req->regex_pattern) : std::string();
    const std::string pool_key_grammar = grammar_enforced ? ("gbnf:" + req->grammar) : std::string();
    const std::string pool_key = tool_enforced ? ConstraintManager::tool_call_key(req->tool_constraint_tools,
                                                                                  req->tool_envelope_open,
                                                                                  req->tool_envelope_close,
                                                                                  req->tool_constraint_xml)
                                 : regex_enforced   ? pool_key_regex
                                 : grammar_enforced ? pool_key_grammar
                                                    : req->json_schema;
    req->constraints = constraints_checkout_(pool_key);

    bool enforced = false;
    if (tool_enforced)
        enforced = req->constraints->prepare_tool_call(req->tool_constraint_tools, req->tool_envelope_open,
                                                       req->tool_envelope_close, model_->tokenizer(),
                                                       /*thinking_open=*/req->in_think_block,
                                                       req->tool_constraint_optional, req->tpl_family,
                                                       req->tool_constraint_parallel,
                                                       req->tool_constraint_bare_args,
                                                       req->tool_constraint_xml);
    if (!enforced && regex_enforced)
        enforced = req->constraints->prepare_regex(req->regex_pattern, model_->tokenizer(),
                                                   /*thinking_open=*/req->in_think_block);
    if (!enforced && grammar_enforced)
        enforced = req->constraints->prepare_grammar(req->grammar, model_->tokenizer(),
                                                     /*thinking_open=*/req->in_think_block);
    if (!enforced && (req->json_mode || !req->json_schema.empty()))
        req->constraints->prepare(req->json_mode, req->json_schema, model_->tokenizer(), req->has_tools,
                                  req->tpl_family, /*thinking_open=*/req->in_think_block);
}

std::shared_ptr<ConstraintManager> Engine::constraints_checkout_(const std::string& json_schema) {
    // Prefer a pooled manager that already classified this schema or grammar
    // (the key is the schema string, or "gbnf:" + the grammar source).
    if (!json_schema.empty()) {
        for (auto it = constraint_pool_.begin(); it != constraint_pool_.end(); ++it) {
            const std::string& cached_gbnf = (*it)->cached_grammar();
            if ((*it)->cached_schema() == json_schema ||
                (!cached_gbnf.empty() && json_schema == "gbnf:" + cached_gbnf)) {
                auto cm = std::move(*it);
                constraint_pool_.erase(it);
                return cm;
            }
        }
    }
    if (!constraint_pool_.empty()) {
        auto cm = std::move(constraint_pool_.back());
        constraint_pool_.pop_back();
        return cm;
    }
    return std::make_shared<ConstraintManager>();
}

void Engine::constraints_return_(std::shared_ptr<ConstraintManager> cm) {
    if (!cm)
        return;
    cm->reset();
    constexpr size_t kMaxConstraintPool = 8;
    if (constraint_pool_.size() < kMaxConstraintPool)
        constraint_pool_.push_back(std::move(cm));
}

bool Engine::preprocess_image(std::span<const uint8_t> data, ImageData& out) {
    return vision_.preprocess(data, out);
}

// Initialization decomposed into sub-phases: init_apply_debug_raw_overrides_ /
// init_resolve_kv_dtype_policy_ / init_resolve_ssm_dtype_ / init_resolve_fp8_prefill_ /
// init_resolve_quant_flags_ / init_compute_max_seq_len_ live in runtime/engine_init_resolver.cpp.

bool Engine::init(std::shared_ptr<Model> model, const EngineConfig& config) {
    if (!model)
        return false;

    // A previous engine on this model handle may have freed the model's source
    // weight tensors to reclaim VRAM (Phase-4b drop): rebuilding weight caches
    // then reads freed memory and poisons the CUDA context (#830). Reject up
    // front; dense models that never drop sources are unaffected.
    if (model->sources_consumed()) {
        IMP_LOG_ERROR(
            "Engine::init: this model handle was already bound to an engine whose "
            "weight caches consumed (freed) the model's source tensors — a second "
            "engine cannot be built on it. Reload the model (imp_model_load) to "
            "create another engine.");
        return false;
    }

    model_ = std::move(model);
    config_ = config;

    // Takes the pending RuntimeConfig stashed by tool main (imp-cli/imp-server)
    // via set_pending_runtime_config(), or a freshly loaded env-seeded default
    // if none was set (library/test embeddings). Every Engine::* method reads
    // runtime_config_ from here on; engine_init_resolver_ helpers mutate it in place.
    runtime_config_ = take_pending_runtime_config();

    // Bridges the documented imp.conf [server]/[paths] keys into EngineConfig
    // (previously parsed into RuntimeConfig but never read, #541). A CLI flag /
    // C-API value can additionally ENABLE a knob (OR), never clobbering an
    // explicit embedder choice; --mmproj (one-shot) overrides imp.conf.
    // `use_prefix_caching` is NOT OR'd: since #758 flipped the RuntimeConfig
    // default to true, OR-ing would silently override an explicit
    // `use_prefix_caching=0` embedder choice. Both shipped tools already pass
    // their own resolved value, so nothing that ships loses caching (#1299/#1314).
    config_.use_green_contexts = config_.use_green_contexts || runtime_config_.server.green_contexts ||
                                 runtime_config_.runtime.prefill_overlap;
    if (config_.prefix_pin_budget_pct == 25)  // EngineConfig default untouched → take imp.conf
        config_.prefix_pin_budget_pct = runtime_config_.server.prefix_pin_budget_pct;
    if (config_.mmproj_path.empty())
        config_.mmproj_path = runtime_config_.paths.mmproj;
    if (config_.vram_budget_mb == 0 && runtime_config_.runtime.vram_budget_mb > 0)
        config_.vram_budget_mb = static_cast<size_t>(runtime_config_.runtime.vram_budget_mb);
    if (config_.kv_fraction == 0.8f)  // EngineConfig default untouched → take imp.conf
        config_.kv_fraction = runtime_config_.vram.kv_fraction;
    if (config_.vram_reserve_floor_pct == 10)
        config_.vram_reserve_floor_pct = runtime_config_.vram.reserve_floor_pct;
    if (config_.library_reserve_mb < 0)
        config_.library_reserve_mb = runtime_config_.vram.library_reserve_mb;
    if (config_.kv_cache_max_blocks == 0 && runtime_config_.kv_cache.max_blocks > 0)
        config_.kv_cache_max_blocks = runtime_config_.kv_cache.max_blocks;
    if (config_.kv_block_size == 0 && runtime_config_.kv_cache.block_size > 0)
        config_.kv_block_size = runtime_config_.kv_cache.block_size;

    // Install the process-wide VRAM budget view BEFORE any sizing runs: every
    // cudaMemGetInfo-based decision below (weight upload gates, cache budgets,
    // KV clamp, workspaces) reads through vram_budget_mem_get_info.
    vram_budget_install(config_.vram_budget_mb);

    // Publishes THIS engine's RuntimeConfig to the process_diag snapshot that
    // leaf kernels read, so exec/ and the kernels agree (a mismatch, e.g.
    // attention.fa2_hd256, walks the FMHA chain to the #654 throw).
    // Ordering matters twice: BEFORE the arch resolvers below (install() writes
    // gemm.cublas_fp16_acc/deterministic_gemm verbatim; resolvers then promote
    // them - installing after would undo arch-specific decisions), and BEFORE
    // the deterministic-gate true-promotion (must follow install so [runtime]
    // deterministic still implies deterministic_gemm).
    process_diag_install(runtime_config_);

    // The deterministic kernel gate lives in process_diag
    // (process_diag_deterministic_gemm()). [runtime] deterministic implies it,
    // but install() only copies deterministic_gemm, so true-promote it here.
    if (runtime_config_.runtime.deterministic || runtime_config_.runtime.deterministic_gemm)
        process_diag_set_deterministic_gemm(true);

    // D1: derive the architecture profile ONCE, before the resolvers below that
    // currently re-derive GDN/SSM/MoE classification inline. The layers are
    // loaded by now (the resolvers read layer().gdn_gate.data directly).
    model_->build_profile();
    {
        const auto& mp = model_->profile();
        const char* av = nullptr;
        switch (mp.attn_variant) {
            case ModelProfile::AttnVariant::GEMMA4_SWA:
                av = "gemma4_swa";
                break;
            case ModelProfile::AttnVariant::GPTOSS_SWA:
                av = "gptoss_swa";
                break;
            case ModelProfile::AttnVariant::NOPE:
                av = "nope";
                break;
            case ModelProfile::AttnVariant::MLA:
                av = "mla";
                break;
            case ModelProfile::AttnVariant::STANDARD:
                av = "standard";
                break;
        }
        IMP_LOG_INFO("ModelProfile: moe=%d gdn=%d ssm=%d hybrid=%d dense=%d attn=%s", mp.is_moe, mp.is_gdn,
                     mp.is_ssm, mp.is_hybrid, mp.is_dense, av);
    }

    init_apply_debug_raw_overrides_();
    init_apply_rope_override_();
    init_resolve_kv_dtype_policy_();
    init_resolve_prefill_graph_();
    init_resolve_ssm_dtype_();
    init_resolve_fp8_prefill_();
    init_resolve_quant_flags_();
    init_resolve_kv_block_size_();

    init_compute_max_seq_len_();

    // Phase-0 VRAM audit harness: lifecycle checkpoints bracket each init
    // sub-phase so the device free-VRAM delta measures that phase's cost with
    // full coverage. Gated, default off.
    if (runtime_config_.diagnostics.vram_audit) {
        MemAccount::instance().set_enabled(true);
        if (!runtime_config_.diagnostics.vram_audit_dump.empty())
            MemAccount::instance().set_dump_path(runtime_config_.diagnostics.vram_audit_dump);
    }
    MemAccount::instance().checkpoint("00_pre_init");
    // Everything already resident before imp allocates anything: CUDA primary
    // context + driver overhead on this WSL2/WDDM box - measured, not assumed,
    // and not imp's memory.
    size_t ctx_baseline_bytes = 0;
    {
        size_t f = 0, tot = 0;
        if (vram_budget_mem_get_info(&f, &tot))
            ctx_baseline_bytes = tot > f ? tot - f : 0;
    }

    // 5% headroom: MoE models (30B Q6_K) need every MiB on 32GB. WSL2/WDDM has
    // ~500 MiB driver overhead, 5% of 32GB = 1.6 GB covers it.
    if (!vram_alloc_.init(kAllocatorHeadroomPct / 100.0f)) {
        IMP_LOG_ERROR("Failed to initialize VRAM allocator");
        return false;
    }
    // Engine-persistent (T2) arena, opened before the first tenant
    // (docs/internals/MEMORY.md A3.3): its Region reserves capacity against
    // everything allocated later, notably the pre-dequant cache build (AUDIT
    // B23). Sized from exact per-tenant demand; the constant is only a floor for tenants not yet migrated.
    {
        // capture_ctx_cap mirrors engine_spec_capture.cpp: the chunk-capture
        // scratch is sized from min(the configured cap, max_seq_len), and is
        // charged only when speculation can actually take that path.
        const int capture_cap = runtime_config_.speculative.capture
                                    ? (config_.max_seq_len > 0
                                           ? std::min(runtime_config_.speculative.capture_ctx_cap,
                                                      config_.max_seq_len)
                                           : runtime_config_.speculative.capture_ctx_cap)
                                    : 0;
        const auto d = exec_t2_demand(*model_, config_.max_seq_len, config_.max_batch_size,
                                      config_.use_fp8_prefill, runtime_config_.attention.mla_absorb,
                                      capture_cap, config_.kv_block_size);
        // The Qwen3-VL tower is engine-lifetime and an arena tenant but uploads
        // during warmup, so its demand must be read off the model's shapes here.
        // Gemma's mmproj tower stays outside the arena for now (docs/audit/SETTLED.md F-12).
        size_t vision_bytes =
            model_->vision_tower ? qwen3vl_vision_arena_bytes(*model_->vision_tower,
                                                              runtime_config_.runtime.vision_max_patches)
                                 : 0;
        if (!config_.mmproj_path.empty())
            vision_bytes += vision_mmproj_arena_bytes(config_.mmproj_path, model_->config_.d_model);
        // The decode batch pool is engine-lifetime and config-sized but
        // allocated after KV sizing, so reserve for it here. with_swa_tables is
        // unknown yet; assume true, guessing high costs nothing, low exhausts the arena.
        const int kv_bs = config_.kv_block_size > 0 ? config_.kv_block_size : 16;
        const size_t batch_pool_bytes = GPUBatchPool::demand_bytes(
            config_.max_batch_size, (config_.max_seq_len + kv_bs - 1) / kv_bs, /*with_swa_tables=*/true);
        // *9/8 for 256-byte alignment padding across the arena's takes (integer
        // identical to t + t/8, just one expression instead of two).
        const size_t cap = std::max(kEngineArenaDefaultBytes, (d.total() + vision_bytes + batch_pool_bytes) * 9 / 8);
        // #1629: the open's outcome must be checked and logged separately for
        // success vs failure, not printed unconditionally before the call.
        // Lazy: reserve the demand, commit as tenants take; the vision tower
        // commits only on the first image, not at load.
        const bool arena_lazy = runtime_config_.vram.lazy_commit && vmm_backend() != nullptr;
        const MemError arena_err = engine_arena_open(arena_lazy ? *vmm_backend() : cuda_malloc_backend(), cap,
                                                     arena_lazy);
        if (arena_err == MemError::Ok) {
            IMP_LOG_INFO("engine arena demand: %s + vision %.1f + batchpool %.2f MiB -> %.1f MiB reserved",
                         d.describe().c_str(), vision_bytes / (1024.0 * 1024.0),
                         batch_pool_bytes / (1024.0 * 1024.0), cap / (1024.0 * 1024.0));
        } else {
            IMP_LOG_WARN(
                "engine arena demand: %s + vision %.1f + batchpool %.2f MiB -> open of "
                "%.1f MiB FAILED; tenants fall back to their own allocations",
                d.describe().c_str(), vision_bytes / (1024.0 * 1024.0), batch_pool_bytes / (1024.0 * 1024.0),
                cap / (1024.0 * 1024.0));
        }
        // Name the two charges already known HERE (context, arena region), not
        // after warmup: lets MemAccount::unattributed_bytes() mean "what imp
        // cannot account for" during init (AUDIT B79). Library figure fills in
        // after warmup measures it.
        // A lazy arena is a tracked pool, not a named charge: naming its
        // reservation would count bytes not backed yet.
        MemAccount::instance().set_named_charges(ctx_baseline_bytes, /*library=*/0,
                                                 engine_arena().lazy() ? 0 : engine_arena().capacity(),
                                                 engine_arena().high_water());
    }
    // T2 slot pool for the conditional graph loop (A7 step 5.3).
    graph_slot_pool_open_for(cuda_malloc_backend(), config_.max_seq_len);
    gemm_init();
    attention_cublas_prewarm();
    // grouped-3x prewarm takes CUTLASS staging + workspace slices from the T2
    // arena so the grouped NVFP4 GEMM never grows mid-capture. Gated on the
    // model having experts, not just sm_120 capability (AUDIT B12): a model
    // without experts never reaches exec/executor_forward_moe*.cu. Workspace measured at 152 320 B (AUDIT B73), not the 512 MiB guess.
    // IMMA prefill scratch (A7 step 8 / AUDIT B13): take it at the charged
    // bound now rather than letting the GEMM climb a staircase of takes later,
    // each intermediate one stranded in a bump arena with no free.
    {
        const auto imma = exec_imma_scratch_shape(*model_, config_.max_seq_len);
        mmq_q8_imma_preallocate(imma.rows, imma.k);
    }
    if (model_->profile().is_moe) {
        gemm_grouped_3x_nvfp4_prewarm();
    } else {
        IMP_LOG_INFO("grouped-3x NVFP4 prewarm skipped: model has no experts");
    }
    // Two different values are called max_batch_size (#1637):
    // EngineConfig::max_batch_size caps ADMISSION here, runtime.max_batch_size
    // truncates the DECODE batch (engine_scheduler.cpp). When decode is the
    // smaller, rows admitted beyond it hold KV and never decode until a head row finishes.
    // Admission clamps to whichever is smaller. Names stay as-is: one is the
    // C-API field, the other a documented imp.conf key; renaming either breaks a published surface.
    int admit_cap = config_.max_batch_size;
    const int decode_cap = runtime_config_.runtime.max_batch_size;
    if (decode_cap > 0 && admit_cap > decode_cap) {
        IMP_LOG_INFO(
            "admission capped at %d to match runtime.max_batch_size (EngineConfig::max_batch_size "
            "is %d). Admitting more than the decode batch can serve parks the extra rows on their "
            "KV until a head row finishes.",
            decode_cap, admit_cap);
        admit_cap = decode_cap;
    }
    scheduler_ = std::make_unique<Scheduler>(admit_cap);
    (void)stream_.create(cudaStreamNonBlocking);
    MemAccount::instance().checkpoint("01_prewarm_gemm");

    // Encoder-only embedder (#836): no executor, KV cache, decoder features,
    // or warmup. Upload weights, dequant into the dedicated encoder workspace;
    // /v1/embeddings drives encoder_embed().
    if (model_->profile().is_encoder) {
        if (!model_->upload_weights_gpu(config_.compute_dtype, stream_, 1ULL << 30)) {
            IMP_LOG_ERROR("encoder: weight upload failed");
            return false;
        }
        auto* ews = new imp::EncoderWorkspace();
        int cap = model_->config_.max_seq_len > 0 ? model_->config_.max_seq_len : 2048;
        if (config_.max_seq_len > 0)
            cap = std::min(cap, config_.max_seq_len);
        if (!imp::encoder_workspace_init(*ews, *model_, cap, stream_)) {
            delete ews;
            return false;
        }
        encoder_ws_storage_ = ews;
        IMP_LOG_INFO("Encoder embedder ready (arch=%s, max_tokens=%d, d=%d)",
                     model_arch_name(model_->config_.arch), cap, model_->config_.d_model);
        return true;
    }

    // Everything from here to the end of warmup is expected to allocate.
    set_alloc_phase(AllocPhase::Planning);
    if (!init_weights()) {
        return false;
    }
    MemAccount::instance().checkpoint("02_weights+decode_cache");
    if (!init_kv_cache()) {
        return false;
    }
    MemAccount::instance().checkpoint("03_kv_cache");
    // Before warmup's graph prewarm: a captured decode step cannot allocate
    // the small-M scratches (#1897). Charged in the T2 arena demand above.
    executor_->allocate_smallm_scratch(stream_);
    if (!init_features())
        return false;
    MemAccount::instance().checkpoint("04_features");
    if (!runtime_config_.runtime.warmup) {
        IMP_LOG_INFO("Warmup SKIPPED (runtime.warmup=false)");
    } else {
        warmup();
    }
    // Last thing before the guard arms: the speculative verify path's one-shot
    // capacity resolutions (A7 step 5.4), which otherwise land mid-serving.
    prewarm_spec_scratch_();
    MemAccount::instance().checkpoint("05_post_warmup");

    // I2: from here on, asking the driver for memory is a defect
    // (docs/internals/MEMORY.md A3.2, AUDIT B8). Debug builds abort on a
    // serving-phase acquisition; release builds count it per tag and log once,
    // since a production server must not die over an accounting bug.
    set_alloc_phase(AllocPhase::Serving);
    // Arm the CUDA-side watermarks too: the phase guard only sees Backend
    // traffic, these see every cudaMallocAsync and every graph-owned
    // allocation regardless of who made it (AUDIT B8).
    MemAccount::instance().arm_steady_state_watermarks();
    // Start the device-used peak sampler, then dump the init table reporting
    // the library reserve the first forward ACTUALLY claimed, not the plan's
    // guessed constant (AUDIT B41/B42) - the plan needs the figure before the
    // forward produces it, so it cannot use this. Falls back to the charged
    // value when warmup was skipped (MXFP4, Gemma-4).
    MemAccount::instance().set_named_charges(ctx_baseline_bytes,
                                             measured_library_reserve_ != SIZE_MAX
                                                 ? measured_library_reserve_
                                                 : engine_internal::library_reserve_charge(
                                                       runtime_config_.vram.library_reserve_mb),
                                             engine_arena().lazy() ? 0 : engine_arena().capacity(),
                                             engine_arena().high_water());
    MemAccount::instance().sampler_start(2000);
    MemAccount::instance().report("init_complete");

    // Model + profile + ssm_state_ are final here and none of the inputs can
    // change afterwards, so the answer is taken once, off the per-step path.
    spec_ngram_model_capable_flag_ = spec_ngram_model_capable_uncached_();

    return true;
}

std::string Engine::generate(const std::string& prompt, int max_tokens, float temperature, float top_p,
                             int top_k, int seed, bool apply_chat_template, float min_p,
                             float repetition_penalty, float frequency_penalty, float presence_penalty) {
    Tokenizer* tok = model_->tokenizer();
    if (!tok) {
        return "";
    }

    std::vector<int32_t> tokens;

    if (apply_chat_template && !chat_template_.is_raw()) {
        std::vector<ChatMessage> messages = {{"user", prompt}};
        if (vision_.has_input() && vision_.is_available()) {
            tokens = chat_template_.apply_with_image(*tok, messages, vision_.num_image_tokens());
        } else {
            tokens = chat_template_.apply(*tok, messages);
        }
        IMP_LOG_INFO("Applied %s chat template (%zu tokens%s)",
                     chat_template_family_name(chat_template_.family()), tokens.size(),
                     vision_.has_input() ? ", with image" : "");
    } else {
        tokens = tok->encode(prompt);
        if (tok->add_bos() && (tokens.empty() || tokens[0] != tok->bos_id())) {
            tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));
        }
    }

    IMP_LOG_INFO("Encoded %zu tokens", tokens.size());
    {
        std::string dump;
        for (size_t i = 0; i < tokens.size() && i < 64; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%d", tokens[i]);
            dump += buf;
            if (i + 1 < tokens.size())
                dump += ", ";
        }
        if (tokens.size() > 64)
            dump += "...";
        IMP_LOG_INFO("Token IDs: [%s]", dump.c_str());
    }

    auto req = std::make_shared<Request>();
    req->id = next_request_id_++;
    req->input_tokens = std::move(tokens);
    req->max_tokens = max_tokens;
    req->temperature = temperature;
    req->top_p = top_p;
    req->top_k = top_k;
    req->seed = seed;
    req->min_p = min_p;
    req->repetition_penalty = repetition_penalty;
    req->frequency_penalty = frequency_penalty;
    req->presence_penalty = presence_penalty;
    req->status = RequestStatus::PENDING;

    scheduler_->add_request(req);

    while (req->status == RequestStatus::PENDING || req->status == RequestStatus::PREFILLING) {
        bool has_work = step();
        if (!has_work)
            break;
    }

    // Decode: try conditional graph loop, fall back to step(). Think budget is
    // enforced device-side in post_decode_step_kernel; penalties applied
    // device-side via apply_penalties_device_count in the graph loop.
    if (req->status == RequestStatus::DECODING && !req->output_tokens.empty() && config_.use_cuda_graphs &&
        !offload_mgr_) {
        int32_t first_token = req->output_tokens.back();
        auto graph_tokens = try_graph_loop_decode(req, first_token, decode_stream());
        if (!graph_tokens.empty()) {
            int32_t last = graph_tokens.back();
            for (int32_t t : graph_tokens)
                track_think_state(*req, t);
            bool hit_stop = should_stop(*req, last);
            if (hit_stop)
                graph_tokens.pop_back();

            for (int32_t t : graph_tokens) {
                req->output_tokens.push_back(t);
            }

            bool done = hit_stop || static_cast<int>(req->output_tokens.size()) >= req->max_tokens;
            if (done) {
                req->status = RequestStatus::FINISHED;
                kv_manager_->free_sequence(req->id);
            }
        }
    }

    // Fallback: per-step decode.
    while (req->status != RequestStatus::FINISHED && req->status != RequestStatus::CANCELLED) {
        bool has_work = step();
        if (!has_work && req->status != RequestStatus::FINISHED && req->status != RequestStatus::CANCELLED) {
            break;
        }
    }

    if (req->output_tokens.empty()) {
        return "";
    }

    vision_.clear_image();

    std::string result = tok->decode(req->output_tokens);
    return result;
}

// prepare_graph_loop: moved to engine_speculative.cpp
// build_graph_config: moved to engine_speculative.cpp
// try_graph_loop_decode: moved to engine_speculative.cpp
// try_launch_async_graph_loop: moved to engine_speculative.cpp

void Engine::add_request(std::shared_ptr<Request> req) {
    if (scheduler_) {
        req->id = next_request_id_++;
        // The join between the two id spaces (#1582): several log sites print
        // `req %d` from the counter above (recurrent-slot, MTP) with only the
        // integer, no Request - the mapping is published once here instead.
        if (!req->trace_id.empty())
            IMP_LOG_INFO("request %s -> engine req %d", req->trace_id.c_str(), req->id);
        // Seed think state from the prompt tail: chat templates (Qwen3.x,
        // DeepSeek-R1) inject `<think>\n` by default, so without this a model
        // closing its empty think block hits should_stop with
        // in_think_block=false and produces a 0-content completion. Scans
        // decoded text of the last input tokens for whichever marker appears last.
        // BOTH flags must be set together: in_think_block alone leaves the
        // ANSWER-headroom budget dead (count_reasoning_tokens counts 0 outside
        // think). imp-server already seeds both since #784 (build_imp_request_);
        // this path covers imp-cli / embedded src/api only.
        Tokenizer* ptok = model_ ? model_->tokenizer() : nullptr;
        if (ptok && !req->input_tokens.empty()) {
            constexpr int kTailScan = 16;  // covers worst case BPE split + slack
            int n = static_cast<int>(req->input_tokens.size());
            int start = std::max(0, n - kTailScan);
            std::string tail_text;
            for (int i = start; i < n; ++i) {
                tail_text += ptok->decode_token(req->input_tokens[i]);
            }
            const think_logic::ThinkSeed seed = think_logic::seed_from_prompt_tail(tail_text);
            if (seed.in_think)
                req->in_think_block = true;
            if (seed.started_in_think)
                req->started_in_think = true;
        }
        // gpt-oss Harmony starts in the analysis (reasoning) channel
        // (<|channel|>analysis<|message|>, no <think> opener for the scan
        // above). Seed think state so answer-headroom counts reasoning from the
        // start and force-closes the channel (<|end|>) before max_tokens exhausts.
        if (harmony_reasoning_) {
            req->started_in_think = true;
            req->in_think_block = true;
        }
        req->prefix_salt = prefix_salt_(*req);
        scheduler_->add_request(std::move(req));
    }
}

}  // namespace imp
