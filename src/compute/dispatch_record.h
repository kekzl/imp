#pragma once

// Records which kernel path a request ACTUALLY ran (#1205): six attention-prefill tiers, five
// MoE-prefill branches and a KV-dtype decode switch each decline silently, so a model taking a
// slower/lower-quality path was invisible. Records the branch that WON at the point it wins,
// from inside the real dispatch - not a predicted path from the pure routing models (which would
// be a third copy of the rules and could disagree with what actually ran).
// Cost: one thread_local store per branch taken, no allocation/logging/sync. Under CUDA-graph
// capture the store happens at capture time, which is correct since replay repeats the captured path.
// Threading: thread_local (graph_diag::g_phase precedent) - the BatchingEngine worker thread is
// the sole caller of Engine::step()/forward, so per-thread state is per-engine state here.

#include "compute/dispatch_paths.h"

namespace imp::dispatch_record {

struct Record {
    AttnPrefillOuter attn_prefill_outer = AttnPrefillOuter::UNSET;
    // Only meaningful when attn_prefill_outer == FMHA_CHAIN.
    AttnPrefillPath attn_prefill_tier = AttnPrefillPath::NONE;
    bool attn_prefill_tier_set = false;

    AttnDecodePath attn_decode = AttnDecodePath::UNSET;

    MoePrefillOuter moe_prefill_outer = MoePrefillOuter::UNSET;
    // Only meaningful when moe_prefill_outer == CUTLASS3X.
    MoePrefillPath moe_prefill_tier = MoePrefillPath::LEGACY;
    bool moe_prefill_tier_set = false;

    bool has_prefill() const { return attn_prefill_outer != AttnPrefillOuter::UNSET; }
    bool has_decode() const { return attn_decode != AttnDecodePath::UNSET; }
};

inline thread_local Record g_record;

inline Record& current() { return g_record; }

inline void set_attn_prefill_outer(AttnPrefillOuter p) { g_record.attn_prefill_outer = p; }

inline void set_attn_prefill_tier(AttnPrefillPath p) {
    g_record.attn_prefill_tier = p;
    g_record.attn_prefill_tier_set = true;
}

inline void set_attn_decode(AttnDecodePath p) { g_record.attn_decode = p; }

inline void set_moe_prefill_outer(MoePrefillOuter p) { g_record.moe_prefill_outer = p; }

inline void set_moe_prefill_tier(MoePrefillPath p) {
    g_record.moe_prefill_tier = p;
    g_record.moe_prefill_tier_set = true;
}

inline void reset() { g_record = Record{}; }

}  // namespace imp::dispatch_record
