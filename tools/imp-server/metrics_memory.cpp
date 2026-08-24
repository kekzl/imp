// Memory metrics for /metrics — invariant I7 (capacity is not occupancy,
// docs/internals/MEMORY.md).
//
// Its own translation unit because handlers_misc.cpp is a grab-bag of unrelated
// endpoints, and adding 44 lines to it pushed the file past the file-size warn
// threshold (627 code LOC against a 600 warn for a normal TU). The gate is a
// proxy for recompile blast radius, and "the misc file grew again" is exactly
// the smell it is meant to surface — so this is a split, not an allowlist entry.

#include "handlers.h"
#include "handlers_internal.h"

#include "api/imp_internal.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "memory/mem_account.h"
#include "exec/executor.h"
#include "runtime/engine.h"

#include <string>

void append_memory_metrics(std::string& out, ServerState& state) {
// Memory: capacity AND occupancy, per tier (invariant I7,
// docs/internals/MEMORY.md). A single "VRAM used" gauge cannot tell a KV
// pool that is 90 % full from one that is 90 % reserved and empty, and both
// capacity questions an operator asks — "can this box take another
// concurrent request?", "is the budget doing anything?" — need the split.
out += "# HELP imp_memory_reserved_bytes Capacity held by a memory tier\n";
out += "# TYPE imp_memory_reserved_bytes gauge\n";
out += "# HELP imp_memory_live_bytes In use inside that tier\n";
out += "# TYPE imp_memory_live_bytes gauge\n";
for (const auto& t : imp::memory_tier_stats()) {
    const std::string tag = std::string("{tier=\"") + t.tier + "\"} ";
    out += "imp_memory_reserved_bytes" + tag + std::to_string(t.reserved) + "\n";
    out += "imp_memory_live_bytes" + tag + std::to_string(t.live) + "\n";
}
// The KV pool is per-engine, so it is added here rather than by the
// process-global snapshot. blocks, not bytes: it is the unit admission
// control actually rations.
if (state.ctx && state.ctx->engine) {
    // MoE expert imbalance (#1548). max(M_e) per launch is what decides grouped
    // GEMM cost: the kernel pads every expert to one M tile, so one hot expert
    // sets it for all of them and the padding is the difference between this
    // ratio and 1. It was computed on the host, used for tiling and dropped, so
    // "would moe.nvfp4_smallM fire on my workload" could only be answered by
    // enabling it and reading a once-per-process INFO line.
    //
    // Scraped live, not written at shutdown like the histogram file: a serving
    // process is exactly the one that could not be asked before.
    if (const auto* ex = state.ctx->engine->executor()) {
        const auto imb = ex->moe_imbalance();
        bool any = false;
        for (const auto& l : imb)
            any = any || l.launches > 0;
        if (any) {
            out += "# HELP imp_moe_expert_imbalance mean max(M_e) over mean rows per expert, per layer\n";
            out += "# TYPE imp_moe_expert_imbalance gauge\n";
            out += "# HELP imp_moe_expert_peak_rows worst max(M_e) seen on that layer\n";
            out += "# TYPE imp_moe_expert_peak_rows gauge\n";
            for (size_t l = 0; l < imb.size(); ++l) {
                if (imb[l].launches == 0)
                    continue;  // dense layer of a hybrid, or not reached yet
                const std::string tag = std::string("{layer=\"") + std::to_string(l) + "\"} ";
                const double ratio = imb[l].mean_rows > 0.0 ? imb[l].mean_max / imb[l].mean_rows : 0.0;
                out += "imp_moe_expert_imbalance" + tag + std::to_string(ratio) + "\n";
                out += "imp_moe_expert_peak_rows" + tag + std::to_string(imb[l].peak_max) + "\n";
            }
        }
    }
    if (const auto* kv = state.ctx->engine->kv_cache()) {
        const int total_blocks = kv->total_blocks();
        const int free_blocks = kv->num_free_blocks();
        out += "# HELP imp_kv_blocks_total KV pool capacity in blocks\n";
        out += "# TYPE imp_kv_blocks_total gauge\n";
        out += "imp_kv_blocks_total " + std::to_string(total_blocks) + "\n";
        out += "# HELP imp_kv_blocks_used KV blocks held by anything — live sequences, the "
               "prefix cache, pins\n";
        out += "# TYPE imp_kv_blocks_used gauge\n";
        out += "imp_kv_blocks_used " + std::to_string(total_blocks - free_blocks) + "\n";
        // The split matters more than the total, and its absence cost me a wrong
        // bug report (#1115): `used` grew by one block per request and looked
        // like a leak, but the server enables prefix caching by default, so
        // free_sequence MOVES the last reference of a hashed block into the
        // cache instead of dropping it. Such a block is occupied AND reclaimable
        // — precisely the distinction invariant I7 exists to make, and a single
        // "used" gauge cannot express it. imp_kv_blocks_live is the one a soak
        // should assert returns to baseline.
        if (auto* mgr = state.ctx->engine->kv_manager()) {
            const int cached = mgr->num_cached_blocks();
            out += "# HELP imp_kv_blocks_cached KV blocks held by the prefix cache (occupied but "
                   "reusable)\n";
            out += "# TYPE imp_kv_blocks_cached gauge\n";
            out += "imp_kv_blocks_cached " + std::to_string(cached) + "\n";
            out += "# HELP imp_kv_blocks_reclaimable Cached blocks eviction can hand back\n";
            out += "# TYPE imp_kv_blocks_reclaimable gauge\n";
            out += "imp_kv_blocks_reclaimable " +
                   std::to_string(mgr->num_reclaimable_cached_blocks()) + "\n";
            out += "# HELP imp_kv_blocks_pinned Cached blocks pinned by cache_control\n";
            out += "# TYPE imp_kv_blocks_pinned gauge\n";
            out += "imp_kv_blocks_pinned " + std::to_string(mgr->num_pinned_blocks()) + "\n";
            out += "# HELP imp_kv_blocks_live KV blocks held by LIVE SEQUENCES only — the number "
                   "that must return to baseline when the load stops\n";
            out += "# TYPE imp_kv_blocks_live gauge\n";
            out += "imp_kv_blocks_live " + std::to_string(total_blocks - free_blocks - cached) +
                   "\n";
            // Promised at admission and not yet written (#1635). Free blocks
            // minus this is what the next request is admitted against, and
            // without it a queue behind a full-looking pool reads as a bug.
            out +=
                "# HELP imp_kv_blocks_reserved KV blocks promised to admitted requests for "
                "their remaining generation, not yet written\n";
            out += "# TYPE imp_kv_blocks_reserved gauge\n";
            out += "imp_kv_blocks_reserved " + std::to_string(mgr->outstanding_reserved_blocks()) + "\n";
        }

        // Speculative decoding (#1321). Without these a spec-decoding test
        // cannot tell whether the drafter ran: the n-gram matcher only fires on
        // repetitive context, so on ordinary prompts drafted stays 0 and the
        // test compares the non-speculative path against itself and passes.
        const auto& sp = state.ctx->engine->spec_stats();
        out += "# HELP imp_spec_drafted_total Draft tokens proposed by speculative decoding\n";
        out += "# TYPE imp_spec_drafted_total counter\n";
        out += "imp_spec_drafted_total " + std::to_string(sp.drafted) + "\n";
        out += "# HELP imp_spec_accepted_total Draft tokens the verify step accepted\n";
        out += "# TYPE imp_spec_accepted_total counter\n";
        out += "imp_spec_accepted_total " + std::to_string(sp.accepted) + "\n";
        out += "# HELP imp_spec_verify_steps_total Verify forwards run\n";
        out += "# TYPE imp_spec_verify_steps_total counter\n";
        out += "imp_spec_verify_steps_total " + std::to_string(sp.verify_steps) + "\n";
        out += "# HELP imp_spec_miss_steps_total Decode steps with no usable draft\n";
        out += "# TYPE imp_spec_miss_steps_total counter\n";
        out += "imp_spec_miss_steps_total " + std::to_string(sp.miss_steps) + "\n";
    }
}
// --vram-budget adherence. own_bytes is this process's allocations since
// engine init — the thing the budget is a cap on — as opposed to
// device-used, which also carries the CUDA context and any neighbour.
{
    const imp::MemBudgetStat b = imp::memory_budget_stat();
    out += "# HELP imp_vram_budget_bytes Installed --vram-budget (0 = uncapped)\n";
    out += "# TYPE imp_vram_budget_bytes gauge\n";
    out += "imp_vram_budget_bytes " + std::to_string(b.budget_bytes) + "\n";
    out += "# HELP imp_vram_own_bytes Device memory allocated by this process since init\n";
    out += "# TYPE imp_vram_own_bytes gauge\n";
    out += "imp_vram_own_bytes " + std::to_string(b.own_bytes) + "\n";
    out += "# HELP imp_vram_own_peak_bytes High water of imp_vram_own_bytes\n";
    out += "# TYPE imp_vram_own_peak_bytes gauge\n";
    out += "imp_vram_own_peak_bytes " + std::to_string(b.own_peak_bytes) + "\n";
}
}
