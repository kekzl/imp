// =============================================================================
// engine_spec_stats.cpp - what speculation counted, and per which drafter
// =============================================================================
//
// Split out of engine_spec_ngram.cpp, which sits at the file-size hard-review
// ceiling: tallying is not what makes that file long, and the per-source split
// added here would have pushed it over.
//
// The counters exist because a speculative-decoding test passes whether or not
// a single token was drafted (#1321): the n-gram matcher only fires on
// repetitive context, so on ordinary prompts the engine logs drafted=0 for
// every request while the test compares the non-speculative path against
// itself. The per-source split exists for the same reason one level down - the
// matcher, the prompt prediction, token recycling and the trained MTP head all
// fill the SAME verify chunk, so an aggregate acceptance rate says nothing
// about the head that costs 0.79 GiB of VRAM.
//
// Exported as imp_spec_mtp_* / imp_spec_ngram_* on /metrics (docs/API.md).
// =============================================================================

#include "core/logging.h"
#include "runtime/engine.h"
#include "runtime/config.h"

namespace imp {

// One verify step's tally, aggregate and per source. Split out of
// step_spec_verify_ because that function sits on the file-size allowlist as a
// ceiling, not an exemption: counting is not what makes it long.
//
// The per-source half is what prices the MTP head. The n-gram/suffix matcher,
// the prompt prediction and token recycling fill the same verify chunk and land
// in the same aggregate, so "drafted 4000, accepted 2600" says nothing about
// the 0.79 GiB the head costs. `from_mtp` is what step_spec_verify_ actually
// used to fill the chunk, not what was configured.
void Engine::spec_stats_record_(bool from_mtp, long long drafted, long long accepted,
                                long long emitted, double wall_ms) noexcept {
    spec_stats_.verify_steps++;
    spec_stats_.drafted += drafted;
    spec_stats_.accepted += accepted;
    spec_stats_.emitted += emitted;
    spec_stats_.verify_wall_ms += wall_ms;
    SpecSourceStats& src = from_mtp ? spec_stats_.mtp : spec_stats_.other;
    src.verify_steps++;
    src.drafted += drafted;
    src.accepted += accepted;
    src.emitted += emitted;
    src.verify_wall_ms += wall_ms;
}

void Engine::log_spec_stats_() const {
    if (spec_stats_.verify_steps + spec_stats_.miss_steps == 0)
        return;
    IMP_LOG_INFO("[spec-ngram] verify_steps=%lld miss_steps=%lld drafted=%lld accepted=%lld "
                 "(%.1f%%) emitted=%lld (%.2f tok/verify, %.2f ms/verify)",
                 spec_stats_.verify_steps, spec_stats_.miss_steps, spec_stats_.drafted,
                 spec_stats_.accepted,
                 spec_stats_.drafted ? 100.0 * spec_stats_.accepted / spec_stats_.drafted : 0.0,
                 spec_stats_.emitted,
                 spec_stats_.verify_steps
                     ? static_cast<double>(spec_stats_.emitted) / spec_stats_.verify_steps
                     : 0.0,
                 spec_stats_.verify_steps ? spec_stats_.verify_wall_ms / spec_stats_.verify_steps
                                          : 0.0);
    // Per source, so a run with the documented MTP pair (mtp_k=2,
    // ngram=false) can be told from one the matcher carried. Same series as
    // /metrics imp_spec_mtp_* / imp_spec_ngram_*.
    if (spec_stats_.mtp.verify_steps + spec_stats_.other.verify_steps > 0) {
        auto pct = [](long long acc, long long dr) { return dr ? 100.0 * acc / dr : 0.0; };
        auto per = [](long long v, long long steps) {
            return steps ? static_cast<double>(v) / steps : 0.0;
        };
        IMP_LOG_INFO("[spec-ngram] by source: mtp verifies=%lld drafted=%lld accepted=%lld (%.1f%%) "
                     "emitted=%lld (%.2f tok/verify) | ngram verifies=%lld drafted=%lld "
                     "accepted=%lld (%.1f%%) emitted=%lld (%.2f tok/verify)",
                     spec_stats_.mtp.verify_steps, spec_stats_.mtp.drafted, spec_stats_.mtp.accepted,
                     pct(spec_stats_.mtp.accepted, spec_stats_.mtp.drafted), spec_stats_.mtp.emitted,
                     per(spec_stats_.mtp.emitted, spec_stats_.mtp.verify_steps),
                     spec_stats_.other.verify_steps, spec_stats_.other.drafted,
                     spec_stats_.other.accepted,
                     pct(spec_stats_.other.accepted, spec_stats_.other.drafted),
                     spec_stats_.other.emitted,
                     per(spec_stats_.other.emitted, spec_stats_.other.verify_steps));
    }
    if (mtp_pool_.tree_branched + mtp_pool_.tree_linear > 0)
        IMP_LOG_INFO("[spec-ngram] mtp tree: branched=%lld linear=%lld (margin gate %.2f)",
                     mtp_pool_.tree_branched, mtp_pool_.tree_linear, runtime_config_.speculative.mtp_tree_margin);
}

}  // namespace imp
