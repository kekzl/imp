#pragma once

// The nine RuntimeConfig sections src/exec reads, lifted from
// runtime/config.h so the hot layer doesn't depend on the top one; live in
// namespace imp::cfg to avoid collisions (imp::KVCache is the paged cache).
// DispatchPolicy is a SNAPSHOT filled after the init resolvers; one writer
// exists after that point (executor_workspace_buffers.cu on
// attention.fmha_prefill_threshold) and every reader is inside src/exec/.
// A future mutation whose readers live outside exec would silently diverge.

#include <cstdint>
#include <string>
#include <vector>

#include "core/config/swa_sizing_mode.h"
#include "core/config/kv_cache.h"
#include "core/config/attention.h"
#include "core/config/moe.h"
#include "core/config/gdn.h"
#include "core/config/gemm.h"
#include "core/config/generation.h"
#include "core/config/speculative.h"
#include "core/config/ffn.h"
#include "core/config/diagnostics.h"

namespace imp {

// The nine sections now live one per header under core/config/; this file
// keeps `#include "core/dispatch_policy.h"` compiling unchanged. A TU that
// needs one section should include that header directly instead.

// The sections exec/ needs, together: a distinct type rather than a handle
// on RuntimeConfig, so core/ never depends on runtime/.
struct DispatchPolicy {
    cfg::KVCache kv_cache;
    cfg::Attention attention;
    cfg::MoE moe;
    cfg::GDN gdn;
    cfg::GEMM gemm;
    cfg::Generation generation;
    cfg::Speculative speculative;
    cfg::FFN ffn;
    cfg::Diagnostics diagnostics;
};

}  // namespace imp
