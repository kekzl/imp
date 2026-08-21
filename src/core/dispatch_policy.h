#pragma once

// The nine RuntimeConfig sections that src/exec reads, lifted out of
// runtime/config.h so the hot layer does not depend on the top one.
//
// Audit finding F-10: config.h was included by 22 files in src/exec/ (85
// translation units transitively) and changed 130 times in six months — the
// highest build cost in the repo. About half that churn lands in sections
// exec/ never reads (Runtime, Vram, Server, Rope); this is what decouples it.
// It is also the durable form of F-1: ProcessDiag exists precisely because
// leaf utilities could not carry a RuntimeConfig.
//
// They live in namespace imp::cfg because un-nesting collides otherwise:
// imp::KVCache is the paged cache in memory/kv_cache.h, and imp-bench has its
// own AttentionConfig. Member names are unchanged, so every existing
// cfg.attention.x still reads exactly the same.
//
// DispatchPolicy is a SNAPSHOT, filled after the init resolvers (engine.cpp:
// resolvers ~851-856, handover ~972). One writer exists after that point —
// executor_workspace_buffers.cu const_casts the derived
// attention.fmha_prefill_threshold — and every reader of that field is inside
// src/exec/, so it writes and reads the same snapshot. Checked, not assumed.
// A future mutation whose readers live OUTSIDE exec would silently diverge.

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

// The nine sections now live one per header under core/config/. This file keeps
// #include "core/dispatch_policy.h" still compiles unchanged. A TU that needs
// one section should include that section's header instead: 21 of the 23 TUs
// that include this one touch two sections or fewer, and one field added here
// costs 137.1 s of rebuild.

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
