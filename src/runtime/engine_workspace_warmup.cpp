// Engine init phase (tail): workspace buffers + banned-token list + warmup.
//
// init_features: green-context overlap, chat-template detection, banned-token list,
// think-token cache, vision pipeline, pinned sample buffer for CUDA graphs, decode-done
// event, DRY penalty buffers. See executor_workspace_*.cu for the per-subsystem builders.
//
// build_banned_token_list: runtime ban-list from RuntimeConfig + model tokenizer, keeping
// stop/EOS/think/Gemma-4 channel markers out of the ban set.
//
// warmup: optional first forward pass to prime cuBLAS + CUDA graph capture. Skipped for
// MXFP4 weights (FP16-cache bypass) and Gemma-4 (algo jitter); resets FP8 KV calibration
// after the synthetic BOS pass.
//
// All three run at engine-init time, colocated as the "init tail": after weights + KV
// cache are up but before the engine becomes ready.

#include "runtime/engine.h"
#include "runtime/config.h"
#include "runtime/engine_internal.h"
#include "runtime/think_stop_logic.h"
#include "compute/sampling.h"
#include "compute/mmq_q8_imma.h"  // mmq_q8_imma_plane_bytes_used
#include "memory/engine_arena.h"
#include "memory/mem_account.h"
#include "memory/plan.h"       // kMeasuredLibraryReserveBytes
#include "memory/vram_query.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace imp {

bool Engine::init_features() {
    const auto& mcfg = model_->config();

    // Green contexts
    if (config_.use_green_contexts) {
        if (!green_ctx_.init(0, config_.green_ctx_prefill_ratio)) {
            IMP_LOG_WARN("Green context init failed — falling back to regular streams");
            // Clear the CUDA error state so it doesn't corrupt subsequent operations. Green context
            // failure on sm_120 consumer GPUs is expected (requires data-center features); without
            // clearing, the stale error causes cublasLtMatmul to fail with CUBLAS_STATUS_INVALID_VALUE.
            cudaGetLastError();
        }
        if (green_ctx_.is_available() && resolve_prefill_chunk_size_() > 0)
            if (executor_->allocate_decode_workspace(stream_, config_.max_batch_size))
                IMP_LOG_INFO("Concurrent prefill/decode overlap enabled");
    }

    // Prefill/decode overlap gates (runtime.prefill_overlap): each DECLINE is logged once
    // (#1755), since a gate that refuses silently is a gate nobody can see. The per-step gate
    // (decode batch >= 2, no spec-verify sharing the CUTLASS activation scratch) lives in step_impl_.
    if (runtime_config_.runtime.prefill_overlap) {
        const char* decline = nullptr;
        if (!green_ctx_.is_available())
            decline = "no stream pair (green-context init failed entirely)";
        else if (!executor_->has_decode_workspace() ||
                 executor_->decode_max_batch() < config_.max_batch_size)
            decline = "decode workspace not allocated at max_batch";
        else if (executor_->model_has_moe())
            decline = "MoE model (expert workspace not per-slot yet — plan phase 2)";
        else if (executor_->has_gguf_nvfp4_overlay())
            decline = "GGUF decode overlay (dp4a/dequant scratch shared with prefill)";
        else if (!executor_->allocate_decode_qscratch(config_.max_batch_size))
            decline = "per-slot quant scratch allocation failed";
        if (!decline) {
            auto slab = engine_arena().take_bytes(SAMPLE_SCRATCH_BYTES);
            if (slab.empty())
                decline = "prefill sample slot allocation failed";
            else
                d_prefill_sample_ = reinterpret_cast<int32_t*>(slab.data());
        }
        overlap_ready_ = (decline == nullptr);
        if (overlap_ready_)
            IMP_LOG_INFO("prefill overlap: ENTERED (engages at decode batch >= 2)");
        else
            IMP_LOG_INFO("prefill overlap: DECLINED — %s; serial prefill path", decline);
    }

    // Chat template
    if (Tokenizer* tok = model_->tokenizer()) {
        auto family = ChatTemplate::detect_family(tok->chat_template_str());
        if (family == ChatTemplateFamily::RAW) {
            family = ChatTemplate::default_family_for_arch(mcfg.arch);
            if (family != ChatTemplateFamily::RAW)
                IMP_LOG_INFO("No chat template in metadata, using %s default for %s",
                             chat_template_family_name(family), model_arch_name(mcfg.arch));
        }
        if (family != ChatTemplateFamily::RAW && !chat_template_.init(family, *tok, tok->chat_template_str()))
            // A failed template init leaves chat_template_ inert, so every /v1/chat/completions
            // request falls back to raw concatenation (no role markers), which looks like a
            // model-quality problem, not a load problem (#1206 marked init() [[nodiscard]]).
            IMP_LOG_WARN(
                "Chat template init failed for family %s — requests will use raw "
                "prompt concatenation (no role markers)",
                chat_template_family_name(family));
    }

    build_banned_token_list();

    // Cache think token IDs for stop-suppression during reasoning.
    // Only treat as think model if <think> is a CONTROL token (from GGUF metadata),
    // not a regular text piece. Nemotron has "<think>" at ID 12 as normal text.
    {
        Tokenizer* ptok = model_->tokenizer();
        if (ptok) {
            int32_t ts = ptok->find_token("<think>");
            int32_t te = ptok->find_token("</think>");
            int vocab = ptok->vocab_size();
            // Accept CONTROL *and* USER_DEFINED token types: Qwen3 GGUFs tag <think>/</think> as
            // USER_DEFINED (type 4); requiring CONTROL only left think_end_id_ at -1, so the think-budget
            // enforcement (force </think> via logit manipulation) could never fire and models thought
            // until max_tokens. Nemotron's "<think>" at ID 12 is type NORMAL text and stays excluded.
            bool accept = think_logic::accept_think_token(
                ts, ptok->has_token_types(), ptok->has_token_types() && ptok->is_special_token(ts),
                ptok->is_added_token(ts), vocab);
            if (accept) {
                think_start_id_ = ts;
                think_end_id_ = te;
            } else if (chat_template_.family() == ChatTemplateFamily::HARMONY) {
                // gpt-oss Harmony: reasoning lives in the analysis channel and closes with <|end|>;
                // the model emits <|channel|>analysis<|message|> itself (no <think> opener, so
                // `accept` is false). Map <|end|> to think_end so the answer-headroom budget can
                // force the analysis -> final channel switch before max_tokens empties the final
                // channel. think_start stays -1: analysis is the initial state, seeded via started_in_think.
                int32_t he = ptok->find_token("<|end|>");
                if (he >= 0) {
                    think_start_id_ = -1;
                    think_end_id_ = he;
                    harmony_reasoning_ = true;
                    // Forced final-channel opener: <|end|> closes analysis, then
                    // <|start|>assistant<|channel|>final<|message|> commits the model to the answer
                    // channel (bare <|end|> lets it re-open analysis). Encode the literal so ids match this tokenizer.
                    harmony_force_seq_ =
                        ptok->encode("<|end|><|start|>assistant<|channel|>final<|message|>");
                    std::string seq;
                    for (int32_t t : harmony_force_seq_) seq += std::to_string(t) + " ";
                    IMP_LOG_INFO("Harmony reasoning: <|end|>=%d, force opener=[ %s]", he, seq.c_str());
                }
            }
            // Build the whitespace-token mask once for any think model (post-</think>/<|end|> grace
            // needs it): a token that decodes to empty/all-whitespace must not count as answer
            // content. Mirrored to device for the conditional-graph loop.
            if (think_end_id_ >= 0) {
                token_is_whitespace_.assign(vocab, 0);
                for (int32_t id = 0; id < vocab; ++id) {
                    if (think_logic::piece_is_whitespace(ptok->decode_token(id)))
                        token_is_whitespace_[id] = 1;
                }
                d_token_is_whitespace_ = static_cast<uint8_t*>(
                    vram_alloc_.allocate(vocab * sizeof(uint8_t),
                                                              "token_is_whitespace"));
                if (d_token_is_whitespace_) {
                    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_token_is_whitespace_,
                                                       token_is_whitespace_.data(),
                                                       vocab * sizeof(uint8_t),
                                                       cudaMemcpyHostToDevice, stream_));
                    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream_));
                }
            }
        }
    }
    {
        // After the think ids: the mask excludes the markers the budget forces.
        std::vector<int32_t> exclude = harmony_force_seq_;
        exclude.push_back(think_start_id_);
        exclude.push_back(think_end_id_);
        Tokenizer* tok = model_->tokenizer();
        stop_mask_.build(banned_token_ids_, tok ? tok->eos_ids() : std::vector<int32_t>{},
                         chat_template_.stop_token_ids(), exclude);
        (void)stop_mask_.device(vram_alloc_, decode_stream());  // upload in the loading phase (I2)
    }

    // Vision
    if (!config_.mmproj_path.empty()) {
        if (!vision_.init(config_.mmproj_path, mcfg.d_model, model_.get(), stream_))
            return false;
    }
    // Qwen3-VL's tower came in with the checkpoint, so there is no mmproj path to key off: its
    // presence IS the signal. A tower that fails to come up is not fatal, the text half of the
    // model works, and refusing to load entirely would be worse than "images are unavailable".
    if (model_->vision_tower) {
        const int budget =
            Qwen3VLPipeline::patch_budget(*model_->vision_tower, runtime_config_.runtime.vision_max_patches);
        // Lazy only when the arena is: without a growable arena a deferred
        // take commits nothing later, it just delays the upload.
        if (!qwen_vision_.init(*model_->vision_tower, budget, engine_arena().lazy())) {
            IMP_LOG_WARN("Qwen3-VL vision tower failed to initialise — continuing text-only");
        } else if (model_->tokenizer()) {
            qwen_image_pad_id_ = model_->tokenizer()->find_token("<|image_pad|>");
            if (qwen_image_pad_id_ < 0)
                IMP_LOG_WARN("Qwen3-VL: no <|image_pad|> in the tokenizer — images unavailable");
        }
    }

    // Pinned sample buffer for CUDA graphs
    if (!h_sample_pinned_) {
        h_sample_pinned_ = PinnedBuffer::acquire(cuda_host_pinned_allocator(), sizeof(int32_t));
        if (h_sample_pinned_.empty())
            demote_graphs_(GraphDemotionReason::PinnedSampleBufUnavailable);
    }
    if (!decode_done_)
        (void)decode_done_.create(cudaEventDisableTiming);

    // Pre-allocate DRY penalty buffers to avoid cudaStreamSynchronize on first
    // use during inference (the lazy-alloc path blocks the decode stream).
    sampling_preallocate_dry(config_.max_seq_len, decode_stream());
    // History-sized penalties (sampling_penalties.cu): per-row token counts,
    // sized like the sampling scratch (max_logit_tokens = max(batch, 8)).
    sampling_preallocate_penalty_counts(std::max(config_.max_batch_size, 8), model_->config().vocab_size);
    // 4096 slots is 32 KiB and four times the server's own --max-logit-bias
    // default, so the fallback path in apply_logit_bias stays unreachable for
    // any request the server accepts (#1617).
    sampling_preallocate_logit_bias(4096);

    return true;
}

void Engine::build_banned_token_list() {
    // Diagnostic bypass: generation.no_ban disables the
    // ban list. Used to bisect Mistral-Small-3.2-NVFP4 long-form repetition
    // (ban vs weight quality).
    if (runtime_config_.generation.no_ban) {
        banned_token_ids_.clear();
        IMP_LOG_WARN("generation.no_ban=true: skipping banned-token list (debug)");
        return;
    }
    banned_token_ids_.clear();
    auto add_if_valid = [this](int32_t id) {
        if (id >= 0) banned_token_ids_.push_back(id);
    };

    // Collect IDs that must NOT be banned: stop tokens, EOS, think tokens,
    // and Gemma-4 channel markers (the model is trained to emit them).
    std::vector<int32_t> keep_ids;
    Tokenizer* tok = model_->tokenizer();
    if (tok) {
        for (int32_t eid : tok->eos_ids()) keep_ids.push_back(eid);
    }
    for (int32_t sid : chat_template_.stop_token_ids()) keep_ids.push_back(sid);
    if (tok) {
        // Harmony (gpt-oss, #547): the model is TRAINED to emit its own structure tokens
        // (<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...).
        // Banning them traps generation in an endless analysis channel.
        for (const char* name : {"<think>", "</think>", "<|think|>", "<|/think|>",
                                  "<|channel>", "<channel|>",
                                  "<|channel|>", "<|message|>", "<|start|>", "<|end|>",
                                  "<|constrain|>"}) {
            int32_t tid = tok->find_token(name);
            if (tid >= 0) keep_ids.push_back(tid);
        }
    }
    auto is_kept = [&](int32_t id) {
        return std::find(keep_ids.begin(), keep_ids.end(), id) != keep_ids.end();
    };

    // Chat template start-of-turn delimiters (never valid in output)
    if (!is_kept(chat_template_.im_start_id()))
        add_if_valid(chat_template_.im_start_id());
    if (!is_kept(chat_template_.start_header_id()))
        add_if_valid(chat_template_.start_header_id());
    if (!is_kept(chat_template_.end_header_id()))
        add_if_valid(chat_template_.end_header_id());

    // Scan vocab for control tokens. Authoritative path uses GGUF token_type
    // metadata; fallback uses heuristic pattern matching on legacy GGUFs.
    if (tok) {
        int vocab_size = tok->vocab_size();
        if (tok->has_token_types()) {
            for (int i = 0; i < vocab_size; i++) {
                if (is_kept(static_cast<int32_t>(i))) continue;
                if (tok->is_control_token(i)) add_if_valid(static_cast<int32_t>(i));
            }
        } else {
            for (int i = 0; i < vocab_size; i++) {
                if (is_kept(static_cast<int32_t>(i))) continue;
                const std::string& t = tok->token_text(i);
                if (t.size() < 3 || t[0] != '<' || t.back() != '>') continue;
                if (t.size() >= 4 && t[1] == '|' && t[t.size() - 2] == '|') {
                    add_if_valid(static_cast<int32_t>(i));
                    continue;
                }
                if (t == "<pad>" || t == "<unk>" || t == "<mask>" || t == "<unused0>" ||
                    t == "<start_of_turn>" || t == "<end_of_turn>" ||
                    t == "<start_of_image>" || t == "<end_of_image>") {
                    add_if_valid(static_cast<int32_t>(i));
                }
            }
        }
    }

    std::sort(banned_token_ids_.begin(), banned_token_ids_.end());
    banned_token_ids_.erase(std::unique(banned_token_ids_.begin(), banned_token_ids_.end()),
                            banned_token_ids_.end());

    if (!banned_token_ids_.empty()) {
        IMP_LOG_INFO("Banned %zu special tokens from generation", banned_token_ids_.size());
        if (tok) {
            constexpr size_t kMaxPrint = 30;
            std::string bl;
            size_t count = std::min(banned_token_ids_.size(), kMaxPrint);
            for (size_t i = 0; i < count; ++i) {
                bl += std::to_string(banned_token_ids_[i]) + "(" + tok->token_text(banned_token_ids_[i]) + ") ";
            }
            if (banned_token_ids_.size() > kMaxPrint)
                bl += "... (+" + std::to_string(banned_token_ids_.size() - kMaxPrint) + " more)";
            IMP_LOG_INFO("  banned: %s", bl.c_str());
        }
    }
    // Upload once, here in the loading phase: the graph paths and the constrained pipeline read
    // this one engine-owned copy. A first-use upload while serving was a counted I2 violation
    // (scripts/check_alloc_interpose.sh).
    (void)banned_tokens_device_(decode_stream());
}

// What the first forward pass actually claimed, against what the plan charged.
//
// The plan needs this number BEFORE the forward that produces it, so it can't measure
// inline: the charge is a constant (kMeasuredLibraryReserveBytes, imp.conf override) that
// AUDIT B41 found wrong in both directions. Measuring here and pinning the exact number is
// stable per model + quant path and invariant to batch/context (M5).
//
// Free function: needs no Engine state beyond its two arguments, keeping engine.h (a
// god-header at its size limit) from growing for a diagnostic. Returns SIZE_MAX if the
// claim could not be measured.
//
// `named_in_window` subtracts what the window swallowed that IS already attributed: the
// IMMA prefill planes the first forward takes and the plan already charged (#1899).
// Without it the window double-counts them as an unattributed library claim across starts.
static size_t measure_library_forward_window(size_t free_before, size_t named_in_window) {
    if (free_before == 0)
        return SIZE_MAX;
    size_t free_after = 0;
    if (!vram_budget_mem_get_info(&free_after, nullptr))
        return SIZE_MAX;
    const size_t window = free_before > free_after ? free_before - free_after : 0;
    return window > named_in_window ? window - named_in_window : 0;
}

// Report the charge against what the plan assumed, and name the value to pin.
//
// The caller takes max(forward_window, whole_init) and caches that: on the NVFP4 path the
// two differ by orders of magnitude, since CUTLASS claims during the phase-02 cache build
// while the window only opens at the warmup forward (AUDIT B79). Runs after the charge is
// decided, so it can recommend the correct number instead of the forward window alone.
static void report_library_reserve(size_t charge, size_t forward_window, size_t whole_init,
                                   int library_reserve_mb) {
    if (charge == SIZE_MAX)
        return;
    const size_t charged = engine_internal::library_reserve_charge(library_reserve_mb);
    const double chg_mib = charged / (1024.0 * 1024.0);
    const double use_mib = charge / (1024.0 * 1024.0);
    const size_t tol = std::max<size_t>(charged / 5, 256ull * 1024 * 1024);
    const size_t diff = charge > charged ? charge - charged : charged - charge;
    if (diff <= tol) {
        IMP_LOG_INFO(
            "library reserve: charged %.0f MiB, this start measured %.0f MiB (within "
            "tolerance)",
            chg_mib, use_mib);
        return;
    }
    IMP_LOG_WARN(
        "library reserve MISMATCH: the plan charged %.0f MiB, this start measured %.0f MiB "
        "(%+.0f MiB, forward window %.0f MiB / whole-init %.0f MiB). %s Pin it for this "
        "model with 'vram.library_reserve_mb = %.0f' in imp.conf: that is the number the "
        "plan and the measurement cache both use. The charge is stable per model and "
        "quant path; the forward WINDOW is not, which is why the two figures above can "
        "differ by orders of magnitude (AUDIT B79).",
        chg_mib, use_mib, use_mib - chg_mib,
        forward_window == SIZE_MAX ? 0.0 : forward_window / (1024.0 * 1024.0), whole_init / (1024.0 * 1024.0),
        charge > charged ? "The plan therefore under-reserved: caches and the KV pool were sized "
                           "against VRAM that the libraries then took."
                         : "The plan therefore over-reserved: that much KV pool was set aside for "
                           "nothing.",
        use_mib);
}

void Engine::warmup() {
    // Skip warmup for MXFP4 models: the warmup forward pass triggers illegal memory access via
    // kernel paths that bypass the FP16 cache and treat raw MXFP4 data as FP16 weights.
    bool has_mxfp4_weights = false;
    for (int i = 0; i < model_->config().n_layers && !has_mxfp4_weights; i++) {
        if (model_->layer(i).wq.qtype == QType::MXFP4)
            has_mxfp4_weights = true;
    }
    if (has_mxfp4_weights) {
        IMP_LOG_INFO("Warmup skipped (MXFP4 model)");
        return;
    }

    // Gemma-4 has outlier-heavy output_norm activations that amplify cuBLAS algo jitter: warming
    // up with BOS-filled buffers pins an algo that produces wrong logits under real inputs and
    // drives decode into backtick/markdown degeneration.
    if (model_->profile().is_gemma4) {
        IMP_LOG_INFO("Warmup skipped (Gemma-4 algo-jitter protection)");
        return;
    }

    Tokenizer* tok = model_->tokenizer();
    int32_t warmup_id = tok ? tok->bos_id() : 1;
    if (warmup_id < 0)
        warmup_id = 1;

    // Split the warmup phase in the VRAM audit: a single checkpoint around the whole phase
    // can't say whether unattributed VRAM is the first forward's library claim, the graph
    // captures, or imp's own lazy workspaces. These three cost one cudaMemGetInfo each, at
    // init only (AUDIT B41).
    MemAccount::instance().checkpoint("05a_pre_warmup_forward");
    size_t warm_free_before = 0;
    vram_budget_mem_get_info(&warm_free_before, nullptr);
    const size_t imma_planes_before = mmq_q8_imma_plane_bytes_used();

    for (int prompt_len : {16, 32}) {
        auto req = std::make_shared<Request>();
        req->id = next_request_id_++;
        req->input_tokens.resize(prompt_len, warmup_id);
        req->max_tokens = 2;
        req->temperature = 0.0f;
        req->ignore_eos = true;
        scheduler_->add_request(req);

        for (int i = 0; i < 8 && req->status != RequestStatus::FINISHED; i++)
            (void)step();

        kv_manager_->free_sequence(req->id);
        reset_ssm_state(req->id);
        while (kv_manager_->evict_cached_block()) {}
        req->status = RequestStatus::CANCELLED;
    }
    MemAccount::instance().checkpoint("05b_post_warmup_forward");
    const size_t imma_planes_in_window = mmq_q8_imma_plane_bytes_used() > imma_planes_before
                                             ? mmq_q8_imma_plane_bytes_used() - imma_planes_before
                                             : 0;
    const size_t forward_window = measure_library_forward_window(warm_free_before, imma_planes_in_window);

    // Which library claims WHEN depends on the model's execution path, so the forward window
    // only sees part of the charge (Q8_0 measures ~7.5 GiB in-forward; NVFP4's CUTLASS claims
    // two phases earlier and measures ~0, AUDIT B79). What imp's own allocations (B78: allocator
    // + async mempool fully named) cannot account for IS the rest (invariant A1.5).
    //
    // This reads the per-pool ledger, which is why note() is no longer gated on
    // --mem-report: with an empty ledger the residual is the whole device, and
    // charging from it collapsed the 35B's KV pool 4096 -> 512 tokens (B80).
    //
    // Take the larger: the window can only see a subset, so a window reading HIGHER means the
    // residual lost something to a pool still growing during warmup. Keep the bigger number
    // rather than under-charge the plan.
    const size_t whole_init = MemAccount::instance().unattributed_bytes();
    measured_library_reserve_ = (forward_window == SIZE_MAX) ? SIZE_MAX
                                                             : std::max(forward_window, whole_init);
    if (forward_window != SIZE_MAX) {
        IMP_LOG_INFO(
            "library reserve: forward window %.0f MiB, whole-init %.0f MiB — charging %.0f MiB "
            "(AUDIT B79/B80)",
            forward_window / (1024.0 * 1024.0), whole_init / (1024.0 * 1024.0),
            measured_library_reserve_ / (1024.0 * 1024.0));
    }
    // Only now is there a number to recommend: everything above decides it, and
    // the warning that names it has to come after, not before (#1746).
    report_library_reserve(measured_library_reserve_, forward_window, whole_init, config_.library_reserve_mb);
    // Remember it, so the NEXT start on this model charges the measured value instead of the
    // constant (AUDIT B49). A write failure is a warning, never a load failure: refusing to
    // serve a model over a cache file would be absurd.
    if (measured_library_reserve_ != SIZE_MAX && !library_reserve_cache_path_.empty()) {
        if (library_reserve_cache_store(library_reserve_cache_path_, library_reserve_key_,
                                        measured_library_reserve_)) {
            // "the next start plans with it" is only true if that path OUTLIVES this process. The
            // supported way to run imp is a container, where the default cache location
            // ($XDG_CACHE_HOME or $HOME/.cache) is ephemeral, so a `docker run --rm` server
            // re-measures forever and keeps charging the constant. Say the condition out loud
            // rather than promise the outcome.
            IMP_LOG_INFO("library reserve: recorded %.0f MiB in %s — the next start on this model "
                         "plans with it IF that path persists (in a container, mount it or set "
                         "vram.library_reserve_cache; otherwise the constant is charged again)",
                         measured_library_reserve_ / (1024.0 * 1024.0),
                         library_reserve_cache_path_.c_str());
        } else {
            IMP_LOG_WARN("library reserve: could not write %s — the next start will charge the "
                         "constant again",
                         library_reserve_cache_path_.c_str());
        }
    }

    for (int i = 0; i < kMaxGraphPoolSize; i++) {
        decode_graph_pool_[i].invalidate();
        // The eager pre-capture warmup step is per-runner state, but what it exists for (cuBLAS
        // autotuning, lazy workspace init) is per-process and already ran via the two warmup
        // requests. Skip it so the first REAL request matches the captured-graph kernel mix of
        // every later one (greedy request-order independence, docs/determinism.md).
        decode_graph_pool_[i].mark_process_warm();
    }
    decode_batch_pool_.reset_upload_cache();
    if (async_graph_runner_.is_setup())
        async_graph_runner_.cleanup();
    if (async_d_block_tables_) {
        IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_));
        async_d_block_tables_ = nullptr;
    }
    if (async_d_block_tables_swa_) {
        IMP_CUDA_CHECK_LOG(cudaFree(async_d_block_tables_swa_));
        async_d_block_tables_swa_ = nullptr;
    }
    async_graph_req_ = nullptr;
    async_pending_tokens_.clear();
    async_pending_cursor_ = 0;
    cudaDeviceSynchronize();
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            IMP_LOG_ERROR("warmup CUDA error: %s", cudaGetErrorString(e));
    }
    // Clear any stale CUDA errors from warmup (e.g. green context reconfigure
    // failure on consumer GPUs: the error propagates to cuBLAS otherwise).
    cudaGetLastError();
    cudaDeviceSynchronize();  // ensure all weight upload/dequant kernels are done

    // Graph prewarm: capture the per-batch-size decode graph pool BEFORE the engine goes ready.
    // Continuous batching visits every batch size on the way up and down, and each never-seen
    // size otherwise pays its capture cost in the first wave of real traffic. One staggered dummy
    // batch walks max_batch_size -> 1 so each pool slot captures now; the last request carries a
    // ~1000-token prompt so max_ctx sits in the 1024 pow2 bucket, keeping the growth-only
    // re-capture trigger (engine_scheduler.cpp) quiet for real requests up to that context. Runs
    // before reset_kv_calibration below so synthetic tokens leave no calibration trace.
    if (runtime_config_.runtime.graph_prewarm && config_.use_cuda_graphs &&
        config_.max_batch_size > 1) {
        const auto t0 = std::chrono::steady_clock::now();
        const int n = std::min(config_.max_batch_size, (int)kMaxGraphPoolSize);
        const int anchor_len = std::min(1000, std::max(16, config_.max_seq_len - 64));
        std::vector<std::shared_ptr<Request>> reqs;
        reqs.reserve(n);
        for (int i = 0; i < n; i++) {
            auto req = std::make_shared<Request>();
            req->id = next_request_id_++;
            // The anchor goes FIRST so its long prefill completes before the short ones start
            // decoding; queued last, the shortest-budget request would finish during the
            // anchor's prefill and the batch would never peak at full size.
            const bool is_anchor = (i == 0);
            req->input_tokens.resize(is_anchor ? anchor_len : 16, warmup_id);
            // Staggered budgets: the batch passes through every size n..1, anchor lives longest.
            // The +4 base keeps the shortest request alive across the chunked-prefill window
            // where late requests still prefill while early ones already decode; a base of 2 let
            // the first request finish before the batch assembled fully, so size n never captured.
            req->max_tokens = is_anchor ? n + 8 : i + 4;
            req->temperature = 0.0f;
            req->ignore_eos = true;
            scheduler_->add_request(req);
            reqs.push_back(std::move(req));
        }
        // n+2 decode steps finish the whole ladder; prefill takes a few more.
        const int step_budget = 4 * n + 32;
        int steps = 0;
        auto unfinished = [&]() {
            for (const auto& r : reqs)
                if (r->status != RequestStatus::FINISHED) return true;
            return false;
        };
        while (unfinished() && steps < step_budget) {
            (void)step();
            steps++;
        }
        int captured = 0;
        std::string missing;
        for (int i = 0; i < n; i++) {
            if (decode_graph_pool_[i].is_ready())
                captured++;
            else
                missing += " " + std::to_string(i + 1);
        }
        for (const auto& r : reqs) {
            kv_manager_->free_sequence(r->id);
            reset_ssm_state(r->id);
            r->status = RequestStatus::CANCELLED;
        }
        while (kv_manager_->evict_cached_block()) {}
        decode_batch_pool_.reset_upload_cache();
        cudaDeviceSynchronize();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        IMP_LOG_INFO("graph prewarm: %d/%d decode graphs captured (%d steps, %.1f s)%s%s",
                     captured, n, steps, dt, missing.empty() ? "" : ", missing sizes:",
                     missing.c_str());
    }
    // Drop FP8 KV calibrated_ flags so the first real prefill re-runs absmax and promotes the
    // per-layer scale via high-water-mark: warmup's synthetic BOS tokens give an unrepresentative
    // K/V absmax. executor_kv_write.cu's high-water-mark logic (FP8 path) keeps the scale
    // monotonically non-decreasing, so warmup's contribution survives if already wider than real
    // prefill, and real prefill widens it further when needed.
    if (executor_)
        executor_->reset_kv_calibration();
    // The prewarm above ran a slot per batch row; the lazy slab hands the pages back so
    // serving starts at zero committed slots.
    IMP_CUDA_CHECK_LOG(cudaDeviceSynchronize());
    trim_recurrent_slots_after_warmup_();
    IMP_LOG_INFO("Warmup complete");
}

}  // namespace imp
