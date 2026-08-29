#include "mtp_auto.h"

#include "core/logging.h"

namespace imp::tools {

void mtp_auto_finalize(RuntimeConfig& cfg, int requested_k, bool head_loaded);

int mtp_auto_request_k(const RuntimeConfig& cfg, int configured_batch) {
    const int configured = cfg.speculative.mtp_k;
    if (configured >= 0)
        return configured;  // explicit: off (0) or a pinned depth
    if (cfg.runtime.deterministic)
        return 0;  // MTP greedy trajectories are not eager-equal
    if (configured_batch != 1)
        return 0;  // concurrent serving: the head binds one request, costs slots
    return kMtpAutoK;
}

int mtp_auto_after_load(RuntimeConfig& cfg, int requested_k, bool head_loaded, int explicit_flag) {
    if (explicit_flag > 0)
        return explicit_flag;  // a tool flag outranks the config
    mtp_auto_finalize(cfg, requested_k, head_loaded);
    // The pending slot was stashed before the load (it gates loader
    // behaviour); re-publish so the engine takes the resolved pair.
    set_pending_runtime_config(cfg);
    // Everything downstream takes the RESOLVED depth: asking to enable a head
    // the checkpoint does not have logs an error for an already-made decision.
    return cfg.speculative.mtp_k;
}

void mtp_auto_finalize(RuntimeConfig& cfg, int requested_k, bool head_loaded) {
    if (cfg.speculative.mtp_k >= 0)
        return;  // explicit configuration, nothing to resolve

    if (requested_k > 0 && head_loaded) {
        cfg.speculative.mtp_k = requested_k;
        // The measured recommendation is the PAIR: with the matcher on, the
        // head drafted 1 token where it drafts 100 with it off (#1796). An
        // operator who set ngram themselves keeps their setting.
        const bool ngram_explicit = cfg.was_set("speculative.ngram");
        if (!ngram_explicit)
            cfg.speculative.ngram = false;
        IMP_LOG_INFO(
            "speculative.mtp_k: auto -> %d with ngram=%s (single-stream run, checkpoint ships an "
            "MTP head; measured +27-30%% on thinking chats, +17-21%% plain decode on "
            "Qwen3.8-27B-NVFP4). Set speculative.mtp_k=0 to opt out.",
            requested_k, cfg.speculative.ngram ? "true (explicitly set)" : "false");
        return;
    }

    cfg.speculative.mtp_k = 0;
    if (requested_k > 0) {
        // Asked for the head, checkpoint does not have one (or it failed to
        // load): say so, and leave the n-gram matcher alone.
        IMP_LOG_INFO("speculative.mtp_k: auto -> off (this checkpoint ships no loadable MTP head)");
    }
}

}  // namespace imp::tools
