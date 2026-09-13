#pragma once

// speculative.mtp_k = -1 ("auto") resolution, shared by imp-cli and imp-server. Cannot live
// in engine_init_resolver: both tools must decide whether to LOAD the MTP head
// (imp_model_load_ex's load_mtp_head, ~0.79 GiB dead VRAM if unused) before an engine exists.
// Resolution runs in two phases around the load and writes back into RuntimeConfig; the engine
// only ever sees a plain 0 or >0. Rule and evidence: cfg::Speculative::mtp_k.

#include "runtime/config.h"

namespace imp::tools {

// Chain depth auto engages on a single-stream run with a head.
inline constexpr int kMtpAutoK = 2;

// Phase 1, before model load: how many rows to ask for, i.e. whether the head is worth
// loading at all. Returns the configured value untouched unless it's the auto sentinel.
// configured_batch is the max_batch_size the tool will pass (0 = engine auto-sizes).
int mtp_auto_request_k(const RuntimeConfig& cfg, int configured_batch);

// Phase 2, after load: installs the resolved pair into cfg and logs the decision once.
// head_loaded is what the load actually produced: a checkpoint without a head must fall back
// to the documented default rather than leave ngram off with nothing drafting.
void mtp_auto_finalize(RuntimeConfig& cfg, int requested_k, bool head_loaded);

// Phase 2 for a caller that stashes the pending runtime config BEFORE the load (imp-cli):
// finalize, re-publish the resolved config so Engine::init takes it, return the depth to
// enable. explicit_flag (--mtp-spec-decode) outranks the config; pass 0 when there is none.
int mtp_auto_after_load(RuntimeConfig& cfg, int requested_k, bool head_loaded, int explicit_flag);

}  // namespace imp::tools
