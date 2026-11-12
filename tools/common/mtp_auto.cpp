#include "mtp_auto.h"

#include "core/logging.h"
#include "model/mtp_head.h"
#include "model/safetensors_loader.h"
#include "runtime/spec_batch_verify_state.h"

#include <algorithm>

namespace imp::tools {

int mtp_auto_request_k(const RuntimeConfig& cfg, int configured_batch) {
    const int configured = cfg.speculative.mtp_k;
    if (configured >= 0)
        return configured;  // explicit: off (0) or a pinned depth
    if (cfg.runtime.deterministic)
        return 0;  // MTP greedy trajectories are not eager-equal
    // Concurrent serving: the head binds one request - unless the batched verify drafts for every
    // request, or speculative.mtp_concurrent lets it draft while one request decodes (phase 2
    // keeps that for GDN hybrids only, the request costs no VRAM before Engine::init).
    if (configured_batch != 1 && !cfg.speculative.batch_verify && !cfg.speculative.mtp_concurrent)
        return 0;
    return kMtpAutoK;
}

MtpAutoServing mtp_auto_serving(const RuntimeConfig& cfg, int configured_batch, bool gdn_hybrid) {
    if (configured_batch == 1 || cfg.speculative.batch_verify)
        return MtpAutoServing::SingleStream;
    return gdn_hybrid && cfg.speculative.mtp_concurrent ? MtpAutoServing::ConcurrentHybrid
                                                        : MtpAutoServing::ConcurrentOther;
}

int mtp_auto_after_load(RuntimeConfig& cfg, int requested_k, bool head_loaded, int explicit_flag,
                        int head_auto_k_cap) {
    if (explicit_flag > 0)
        return explicit_flag;  // a tool flag outranks the config
    mtp_auto_finalize(cfg, requested_k, head_loaded, head_auto_k_cap);
    // The pending slot was stashed before the load (it gates loader
    // behaviour); re-publish so the engine takes the resolved pair.
    set_pending_runtime_config(cfg);
    // Everything downstream takes the RESOLVED depth: asking to enable a head
    // the checkpoint does not have logs an error for an already-made decision.
    return cfg.speculative.mtp_k;
}

namespace {
// A concurrent GDN-hybrid server keeps the KV pool: drafts score LM-head rows [0, 98304) and the NVFP4
// draft copy holds only those. Drafts past the first per bound request stay while factor rows replace
// their spare slots (speculative.mtp_batch_verify_factor_row, 3.22 MiB vs 79.5 MiB per row); else they go.
void mtp_concurrent_pool_defaults(RuntimeConfig& cfg, MtpAutoServing serving) {
    if (cfg.speculative.mtp_k <= 0 || serving != MtpAutoServing::ConcurrentHybrid)
        return;
    // KV host spill restores an evicted prefix by H2D, so the second draft per row stays.
    const bool spill = cfg.kv_cache.host_spill_mb > 0;
    const bool k = !spill && !cfg.speculative.mtp_batch_verify_factor_row &&
                   !cfg.was_set("speculative.mtp_batch_verify_k");
    if (k)
        cfg.speculative.mtp_batch_verify_k = 1;
    // Rows [0, 98304) hold 99.94 % of Qwen3.8 agent output tokens.
    const bool vocab = !cfg.was_set("speculative.mtp_draft_vocab");
    if (vocab)
        cfg.speculative.mtp_draft_vocab = 98304;
    // Qwen3.8-27B, 1 chat: prefix copy (270 MiB, 786 KV blocks) 204.26 tok/s, FP8 head 198.04, full copy
    // (682 MiB, 1990 blocks) 192.68. The full copy (mtp_draft_vocab=0 set) stays off.
    const bool head = !cfg.was_set("speculative.mtp_nvfp4_head") && cfg.speculative.mtp_draft_vocab <= 0;
    if (head)
        cfg.speculative.mtp_nvfp4_head = false;
    if (head || k || vocab)
        IMP_LOG_INFO(
            "speculative: concurrent server -> mtp_nvfp4_head=%s, mtp_draft_vocab=%d, mtp_batch_verify_k=%d "
            "(KV pool kept for concurrent chats; set a key to override)",
            cfg.speculative.mtp_nvfp4_head ? "true" : "false", cfg.speculative.mtp_draft_vocab,
            cfg.speculative.mtp_batch_verify_k);
}
}  // namespace

void mtp_auto_finalize(RuntimeConfig& cfg, int requested_k, bool head_loaded, int head_auto_k_cap,
                       MtpAutoServing serving) {
    if (cfg.speculative.mtp_k >= 0) {
        mtp_concurrent_pool_defaults(cfg, serving);
        mtp_verify_depth_follows_k(cfg);
        return;  // explicit depth, nothing else to resolve
    }

    if (requested_k > 0 && head_loaded && serving == MtpAutoServing::ConcurrentOther) {
        // Not a GDN hybrid: batch > 1 round-robin verifies with the n-gram matcher, which the pair
        // would turn off. Keep the documented concurrent default.
        cfg.speculative.mtp_k = 0;
        IMP_LOG_INFO(
            "speculative.mtp_k: auto -> off (concurrent server on a model that is not a GDN hybrid). "
            "Set speculative.mtp_k=2 with speculative.ngram=false to force it.");
        return;
    }
    if (requested_k > 0 && head_loaded && head_auto_k_cap < 0) {
        cfg.speculative.mtp_k = 0;
        IMP_LOG_INFO(
            "speculative.mtp_k: auto -> off (Qwen3.8-Flash-Next k=1 decodes 53.15 vs 65.31 tok/s spec off, "
            "-18.6 %%, #2272; head holds 2400 MiB). Set speculative.mtp_k=1 to force it.");
        return;
    }
    if (requested_k > 0 && head_loaded) {
        // A concurrent GDN hybrid drafts what its bound verify stages per request (up to 7 with factor rows).
        if (serving == MtpAutoServing::ConcurrentHybrid && mtp_bound_verify_config_rows(cfg) > 0)
            requested_k = std::max(requested_k, mtp_bound_verify_top_depth(cfg));
        if (head_auto_k_cap > 0 && requested_k > head_auto_k_cap)
            requested_k = head_auto_k_cap;  // the head's own auto depth (mtp_auto_k_cap)
        cfg.speculative.mtp_k = requested_k;
        // The measured recommendation is the PAIR: with the matcher on, the
        // head drafted 1 token where it drafts 100 with it off (#1796). An
        // operator who set ngram themselves keeps their setting.
        const bool ngram_explicit = cfg.was_set("speculative.ngram");
        if (!ngram_explicit)
            cfg.speculative.ngram = false;
        IMP_LOG_INFO(
            "speculative.mtp_k: auto -> %d with ngram=%s (%s, checkpoint ships an "
            "MTP head; measured +27-30%% on thinking chats, +17-21%% plain decode on "
            "Qwen3.8-27B-NVFP4). Set speculative.mtp_k=0 to opt out.",
            requested_k, cfg.speculative.ngram ? "true (explicitly set)" : "false",
            serving == MtpAutoServing::ConcurrentHybrid
                ? "concurrent server: drafts while one request decodes, speculative.mtp_concurrent"
                : "single-stream run");
        mtp_concurrent_pool_defaults(cfg, serving);
        mtp_verify_depth_follows_k(cfg);
        return;
    }

    cfg.speculative.mtp_k = 0;
    if (requested_k > 0) {
        // Asked for the head, checkpoint does not have one (or it failed to
        // load): say so, and leave the n-gram matcher alone.
        IMP_LOG_INFO("speculative.mtp_k: auto -> off (this checkpoint ships no loadable MTP head)");
    }
}

void mtp_attach_external_head(Model& model, const RuntimeConfig& cfg, int requested_k) {
    if (requested_k <= 0 || cfg.speculative.mtp_head.empty() || (model.mtp_ && model.mtp_->loaded))
        return;
    (void)attach_mtp_head_file(model, cfg.speculative.mtp_head);
}

}  // namespace imp::tools
