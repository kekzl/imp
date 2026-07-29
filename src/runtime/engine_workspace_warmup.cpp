// Engine init phase (tail): workspace buffers + banned-token list + warmup.
//
// init_features:
//   Initialises green-context overlap, chat-template detection, banned-
//   token list, think-token cache, vision pipeline, pinned sample buffer
//   for CUDA graphs, decode-done event, and pre-allocates DRY penalty
//   buffers. See executor_workspace_*.cu for the per-subsystem workspace
//   builders that this method drives indirectly.
//
// build_banned_token_list:
//   Constructs the runtime ban-list from RuntimeConfig + model tokenizer.
//   Keeps stop/EOS/think/Gemma-4 channel markers out of the ban set.
//
// warmup:
//   Optional first forward pass to prime cuBLAS + CUDA graph capture.
//   Skipped for MXFP4 weights (FP16-cache bypass) and Gemma-4 (algo
//   jitter). Resets FP8 KV calibration after the synthetic BOS pass.
//
// All three execute at engine-init time and are colocated here as the
// "init tail" — after weights + KV cache are up but before the engine
// becomes ready.
//
// Extracted from engine.cpp in Phase 4 of the architecture refactor
// roadmap. Bodies byte-identical.

#include "runtime/engine.h"
#include "runtime/config.h"
#include "runtime/think_stop_logic.h"
#include "compute/sampling.h"
#include "core/logging.h"

#include <cuda_runtime.h>
#include <algorithm>
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
            // Clear the CUDA error state so it doesn't corrupt subsequent operations.
            // Green context failure on sm_120 consumer GPUs is expected (requires
            // data-center features). Without clearing, the stale error causes
            // cublasLtMatmul to fail with CUBLAS_STATUS_INVALID_VALUE.
            cudaGetLastError();
        }
        if (green_ctx_.is_available() && resolve_prefill_chunk_size_() > 0)
            if (executor_->allocate_decode_workspace(stream_, config_.max_batch_size))
                IMP_LOG_INFO("Concurrent prefill/decode overlap enabled");
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
        if (family != ChatTemplateFamily::RAW)
            chat_template_.init(family, *tok, tok->chat_template_str());
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
            // Accept CONTROL *and* USER_DEFINED token types: Qwen3 GGUFs tag
            // <think>/</think> as USER_DEFINED (type 4), and requiring CONTROL
            // left think_end_id_ at -1 — the think-budget enforcement
            // (force </think> via logit manipulation) could then never fire,
            // so models thought until max_tokens (empty content under
            // json_mode/short budgets). Nemotron's "<think>" at ID 12 is type
            // NORMAL text and stays excluded.
            bool accept = think_logic::accept_think_token(
                ts, ptok->has_token_types(), ptok->has_token_types() && ptok->is_special_token(ts),
                ptok->is_added_token(ts), vocab);
            if (accept) {
                think_start_id_ = ts;
                think_end_id_ = te;
            } else if (chat_template_.family() == ChatTemplateFamily::HARMONY) {
                // gpt-oss Harmony: reasoning lives in the analysis channel and
                // closes with <|end|>; the model emits <|channel|>analysis
                // <|message|> itself (no <think> opener, so `accept` is false).
                // Map <|end|> to think_end so the answer-headroom budget can
                // force the analysis -> final channel switch when reasoning
                // would otherwise consume all of max_tokens and leave the final
                // channel (the answer) empty (finish=length). think_start stays
                // -1: analysis is the initial state, seeded via started_in_think.
                int32_t he = ptok->find_token("<|end|>");
                if (he >= 0) {
                    think_start_id_ = -1;
                    think_end_id_ = he;
                    harmony_reasoning_ = true;
                    // Forced final-channel opener: <|end|> closes analysis, then
                    // <|start|>assistant<|channel|>final<|message|> commits the
                    // model to the answer channel (forcing <|end|> alone lets it
                    // re-open analysis). Encode the literal so the exact ids
                    // (incl. role/channel-name pieces) match this tokenizer.
                    harmony_force_seq_ =
                        ptok->encode("<|end|><|start|>assistant<|channel|>final<|message|>");
                    std::string seq;
                    for (int32_t t : harmony_force_seq_) seq += std::to_string(t) + " ";
                    IMP_LOG_INFO("Harmony reasoning: <|end|>=%d, force opener=[ %s]", he, seq.c_str());
                }
            }
            // Build the whitespace-token mask once for any think model (the
            // post-</think>/<|end|> grace needs it). A token that decodes to
            // empty/all-whitespace must not count as answer content. Mirror to
            // device for the conditional-graph loop.
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

    // Vision
    if (!config_.mmproj_path.empty()) {
        if (!vision_.init(config_.mmproj_path, mcfg.d_model, model_.get(), vram_alloc_, stream_))
            return false;
    }

    // Pinned sample buffer for CUDA graphs
    if (!h_sample_pinned_) {
        cudaError_t err = cudaHostAlloc(&h_sample_pinned_, sizeof(int32_t), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("cudaHostAlloc for sample buffer failed: %s", cudaGetErrorString(err));
            if (config_.use_cuda_graphs)
                config_.use_cuda_graphs = false;
            h_sample_pinned_ = nullptr;
        }
    }
    if (!decode_done_)
        (void)decode_done_.create(cudaEventDisableTiming);

    // Pre-allocate DRY penalty buffers to avoid cudaStreamSynchronize on first
    // use during inference (the lazy-alloc path blocks the decode stream).
    sampling_preallocate_dry(config_.max_seq_len, decode_stream());

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
        // Harmony (gpt-oss, #547): the model is TRAINED to emit its own
        // structure tokens (<|channel|>analysis<|message|>...<|end|>
        // <|start|>assistant<|channel|>final<|message|>...). Banning them
        // traps generation in an endless analysis channel.
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

    // Deduplicate
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
}

void Engine::warmup() {
    // Skip warmup for MXFP4 models — the warmup forward pass triggers
    // illegal memory access due to kernel paths that bypass the FP16 cache
    // and attempt to use raw MXFP4 data as FP16 weights.
    bool has_mxfp4_weights = false;
    for (int i = 0; i < model_->config().n_layers && !has_mxfp4_weights; i++) {
        if (model_->layer(i).wq.qtype == QType::MXFP4)
            has_mxfp4_weights = true;
    }
    if (has_mxfp4_weights) {
        IMP_LOG_INFO("Warmup skipped (MXFP4 model)");
        return;
    }

    // Gemma-4 has outlier-heavy output_norm activations that amplify cuBLAS
    // algo jitter — warming up with BOS-filled buffers pins an algo that
    // produces wrong logits under real inputs and drives decode into
    // backtick/markdown degeneration. IMP_NO_WARMUP=1 was the manual
    // mitigation; make it automatic for the arch.
    if (model_->profile().is_gemma4) {
        IMP_LOG_INFO("Warmup skipped (Gemma-4 algo-jitter protection)");
        return;
    }

    Tokenizer* tok = model_->tokenizer();
    int32_t warmup_id = tok ? tok->bos_id() : 1;
    if (warmup_id < 0)
        warmup_id = 1;

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

    for (int i = 0; i < kMaxGraphPoolSize; i++) {
        decode_graph_pool_[i].invalidate();
        // The eager pre-capture warmup step is per-runner state, but what it
        // exists for (cuBLAS autotuning, lazy workspace init) is per-process
        // and just ran via the two warmup requests. Skip it so the first REAL
        // request executes the same captured-graph kernel mix as every later
        // one — greedy request-order independence (see docs/determinism.md).
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
    // failure on consumer GPUs — the error propagates to cuBLAS otherwise).
    cudaGetLastError();
    cudaDeviceSynchronize();  // ensure all weight upload/dequant kernels are done
    // Drop FP8 KV calibrated_ flags so the first real prefill re-runs absmax
    // and promotes the per-layer scale via high-water-mark. Warmup uses
    // synthetic BOS tokens whose K/V absmax is unrepresentative; without this
    // reset, Llama-3.2-3B with --kv-fp8 degenerated to " France, and, 2008,
    // 201, 201, …" within 30 tokens. The high-water-mark logic in
    // executor_kv_write.cu (FP8 path) keeps the scale monotonically
    // non-decreasing, so warmup's contribution survives if it was already
    // wider than real prefill (Qwen3 case), and real prefill widens it
    // further when needed (Llama case).
    if (executor_)
        executor_->reset_kv_calibration();
    IMP_LOG_INFO("Warmup complete");
}

}  // namespace imp
