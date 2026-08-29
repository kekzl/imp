#pragma once

// speculative.mtp_k = -1 ("auto") resolution, shared by imp-cli and
// imp-server.
//
// This cannot live in engine_init_resolver: both tools must decide whether to
// LOAD the MTP head (imp_model_load_ex's load_mtp_head) before an engine
// exists, and the head is ~0.79 GiB of dead VRAM when nothing drafts with it.
// So the resolution runs in two phases around the load, and writes the
// resolved values back into the RuntimeConfig the engine then takes - the
// engine only ever sees a plain 0 or >0, exactly as before.
//
// The rule and its evidence are documented at cfg::Speculative::mtp_k.

#include "runtime/config.h"

namespace imp::tools {

// Chain depth auto engages on a single-stream run with a head.
inline constexpr int kMtpAutoK = 2;

// Phase 1, before the model load: how many rows to ask for, i.e. whether the
// head is worth loading at all. Returns the configured value untouched unless
// it is the auto sentinel. `configured_batch` is the max_batch_size the tool
// will pass to the engine (0 = engine auto-sizes = concurrent serving).
int mtp_auto_request_k(const RuntimeConfig& cfg, int configured_batch);

// Phase 2, after the load: install the resolved pair into `cfg` and log the
// decision once. `head_loaded` is what the load actually produced - a
// checkpoint without a head must fall back to the documented default rather
// than leave ngram off with nothing drafting.
void mtp_auto_finalize(RuntimeConfig& cfg, int requested_k, bool head_loaded);

// Phase 2 for a caller that stashes the pending runtime config BEFORE the
// load (imp-cli): finalize, re-publish the resolved config so Engine::init
// takes it, and return the depth to enable. `explicit_flag` is a tool flag
// that outranks the config (--mtp-spec-decode); pass 0 when there is none.
int mtp_auto_after_load(RuntimeConfig& cfg, int requested_k, bool head_loaded, int explicit_flag);

}  // namespace imp::tools
