#pragma once

// speculative.mtp_k = -1 ("auto") resolution, shared by imp-cli and imp-server. Cannot live
// in engine_init_resolver: both tools must decide whether to LOAD the MTP head
// (imp_model_load_ex's load_mtp_head, ~0.79 GiB dead VRAM if unused) before an engine exists.
// Resolution runs in two phases around the load and writes back into RuntimeConfig; the engine
// only ever sees a plain 0 or >0. Rule and evidence: cfg::Speculative::mtp_k.

#include "model/model.h"
#include "runtime/config.h"

#include <optional>

namespace imp::tools {

// Chain depth auto engages on a single-stream run with a head.
inline constexpr int kMtpAutoK = 2;

// The serving regime auto resolves for. A concurrent GDN hybrid takes the head too
// (speculative.mtp_concurrent): batch > 1 never verifies there, so ngram=false costs nothing.
enum class MtpAutoServing { SingleStream, ConcurrentHybrid, ConcurrentOther };
MtpAutoServing mtp_auto_serving(const RuntimeConfig& cfg, int configured_batch, bool gdn_hybrid);

// Engine batch of a batch-1 GDN hybrid auto drafts on: the bound verify stages its rows only at engine
// batch 2, so this returns 2 and pins runtime.max_batch_size=1 (admission, decode and KV ceiling stay
// at one chat). Else configured_batch. Qwen3.8-27B agent replay, 1 chat: 69.28 -> 205.58 tok/s.
inline constexpr int kMtpBoundVerifyEngineBatch = 2;
int mtp_auto_engine_batch(RuntimeConfig& cfg, int configured_batch, bool gdn_hybrid, int requested_k,
                          bool head_loaded, int head_auto_k_cap = 0);

// Top of every server model load: the phases below write resolved keys into cfg, and a reload (resume,
// idle unload, swap) would read them as explicit. First call snapshots cfg, later calls restore it.
void mtp_auto_restore_startup(RuntimeConfig& cfg, std::optional<RuntimeConfig>& startup);

// Phase 1, before model load: how many rows to ask for, i.e. whether the head is worth
// loading at all (host mmap only; the upload is Engine::init's). Returns the configured value
// untouched unless it's the auto sentinel. configured_batch is the max_batch_size the tool will
// pass (0 = engine auto-sizes).
int mtp_auto_request_k(const RuntimeConfig& cfg, int configured_batch);

// Phase 2, after load: installs the resolved pair into cfg and logs the decision once.
// head_loaded is what the load actually produced: a checkpoint without a head must fall back
// to the documented default rather than leave ngram off with nothing drafting.
// head_auto_k_cap > 0 (mtp_auto_k_cap of the loaded head) caps the auto depth only, < 0 turns
// auto off for that head; explicit settings pass through. ConcurrentOther declines.
void mtp_auto_finalize(RuntimeConfig& cfg, int requested_k, bool head_loaded, int head_auto_k_cap = 0,
                       MtpAutoServing serving = MtpAutoServing::SingleStream);

// Phase 2 for a caller that stashes the pending runtime config BEFORE the load (imp-cli):
// finalize, re-publish the resolved config so Engine::init takes it, return the depth to
// enable. explicit_flag (--mtp-spec-decode) outranks the config; pass 0 when there is none.
int mtp_auto_after_load(RuntimeConfig& cfg, int requested_k, bool head_loaded, int explicit_flag,
                        int head_auto_k_cap = 0);

// Between load and phase 2: speculative.mtp_head attaches an external head when MTP was requested
// (requested_k > 0) and the checkpoint brought none. No-op otherwise.
void mtp_attach_external_head(Model& model, const RuntimeConfig& cfg, int requested_k);

}  // namespace imp::tools
