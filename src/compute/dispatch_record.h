#pragma once

// Which kernels a request ACTUALLY ran (#1205).
//
// imp resolves a model to a specific set of kernels through several chains —
// six attention-prefill tiers, five MoE-prefill branches, a KV-dtype decode
// switch — and every one of them declines by returning `false` with no log.
// The result was that a model silently taking a slower or lower-quality path
// was invisible: the audit's "no resolved-path dump exists, so every future
// routing regression is invisible" finding.
//
// This records the branch that WON, at the point it wins, from inside the real
// dispatch. That is the whole design constraint: a *predicted* path (running
// the pure routing models from select_attn_prefill_path() at init) would be a
// third copy of the routing rules and could be wrong exactly when it matters.
// What is recorded here cannot disagree with what ran, because it is set by the
// code that ran.
//
// Cost: one thread_local store per branch taken, on paths that are already
// doing a kernel launch. No allocation, no logging, no synchronisation. Under
// CUDA-graph capture the store happens at capture time — which is the correct
// moment, since replay repeats exactly the captured path.
//
// Threading: thread_local, following the graph_diag::g_phase precedent. The
// BatchingEngine worker thread is the sole caller of Engine::step()/forward, so
// per-thread state is per-engine state here; a second inference thread simply
// gets its own record rather than corrupting a shared one.

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
