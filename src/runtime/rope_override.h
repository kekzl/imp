#pragma once

#include "runtime/config.h"

namespace imp {

struct ModelConfig;

// Apply the imp.conf [rope] runtime override to a loaded ModelConfig (see
// RuntimeConfig::Rope). Bumps max_seq_len to factor × orig_ctx. Returns
// false when off or refused (LongRoPE/llama3 per-dim tables, MLA, NoPE, bad
// factor). Free function: host-only tests can drive it without a GPU.
[[nodiscard]] bool apply_rope_override(ModelConfig& mcfg, const RuntimeConfig::Rope& rope);

}  // namespace imp
