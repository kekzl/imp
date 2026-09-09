#include "runtime/plan_shadow.h"
#include "runtime/vram_budget.h"
#include "core/logging.h"

#include <cstdio>

namespace imp {

namespace {
constexpr double kMiB = 1024.0 * 1024.0;
}

PlanInput shadow_plan_input(const ShadowPlanProbe& probe) {
    PlanInput in;

    // Weights and the CUDA context are already resident at this point, so the
    // budget being distributed is what is left — charging them again would
    // double-count. This is the one place the shadow plan is not the shape it
    // will have in step 6, where the plan runs BEFORE the upload and charges
    // everything from a clean slate.
    in.budget_bytes = probe.distributable_bytes;
    in.context_bytes = 0;
    in.model.weight_bytes = 0;

    in.model.n_kv_layers = probe.n_kv_layers;
    in.model.weight_cache_bytes = probe.weight_cache_demand;
    in.model.mandatory_cache_bytes = probe.mandatory_cache_bytes;

    in.features.ssm_state_bytes = probe.ssm_state_bytes;
    in.features.recurrent_snapshot_bytes = probe.recurrent_snapshot_bytes;
    in.features.spec_decode_bytes = probe.spec_decode_bytes;
    in.features.residual_ring_bytes = probe.residual_ring_bytes;
    // features.vision_tower_bytes stays 0 deliberately: the tower is an engine
    // ARENA tenant and the arena is already open when this probe is built, so
    // its bytes are outside distributable_bytes. Charging it here would
    // double-count it. The note at the bottom of the report says so.
    in.features.n_swa_layers = probe.n_swa_layers;
    in.features.swa_live_tokens = probe.swa_live_tokens;

    in.limits.max_batch_size = probe.max_batch_size;
    in.limits.max_seq_len = probe.max_seq_len;
    in.limits.kv_block_size = probe.kv_block_size;
    in.limits.kv_block_bytes_per_layer = probe.kv_block_bytes_per_layer;
    in.limits.min_kv_tokens = probe.min_kv_tokens;

    in.engine_persistent_bytes = probe.engine_persistent_bytes;
    in.forward_scratch_bytes = 0;  // step 4 introduces the stack that measures it

    in.library.bytes = probe.library_reserve_bytes;
    in.library.source = "vram.library_reserve_mb";
    return in;
}

std::string shadow_plan_report(const ShadowPlanProbe& probe, const PlanResult& shadow,
                               int live_kv_blocks) {
    const size_t per_block =
        probe.kv_block_bytes_per_layer *
        static_cast<size_t>(probe.n_kv_layers > probe.n_swa_layers
                                ? probe.n_kv_layers - probe.n_swa_layers
                                : probe.n_kv_layers);

    std::string out;
    char line[512];
    auto emit = [&](const char* fmt, auto... args) {
        // Without arguments `fmt` is still a runtime pointer as far as the
        // compiler can tell, so a stray '%' in it would read operands that were
        // never passed (-Wformat-security). Print it as data in that case.
        if constexpr (sizeof...(args) == 0)
            std::snprintf(line, sizeof(line), "%s", fmt);
        else
            std::snprintf(line, sizeof(line), fmt, args...);
        out += line;
        out += '\n';
    };

    emit("memory plan (A7 step 2 — APPLIED: the KV block count below is what the pool uses):");
    emit("  distributable            %8.0f MiB", probe.distributable_bytes / kMiB);
    emit("  weight-cache demand      %8.0f MiB  (same figure the live pass used)",
         probe.weight_cache_demand / kMiB);
    emit("  library reserve          %8.0f MiB  (%s)", probe.library_reserve_bytes / kMiB,
         probe.library_reserve_bytes ? "charged by both passes since #1109" : "disabled");
    if (probe.ssm_state_bytes)
        emit("  SSM/GDN state            %8.0f MiB  (%d slots)", probe.ssm_state_bytes / kMiB,
             probe.max_batch_size);
    if (probe.recurrent_snapshot_bytes)
        emit("  recurrent snapshots      %8.0f MiB  (server.recurrent_snapshot_mb)",
             probe.recurrent_snapshot_bytes / kMiB);
    if (probe.spec_decode_bytes)
        emit("  speculative staging      %8.0f MiB", probe.spec_decode_bytes / kMiB);
    if (probe.residual_ring_bytes)
        emit("  residual FP16 ring       %8.0f MiB", probe.residual_ring_bytes / kMiB);
    if (probe.engine_persistent_bytes)
        emit("  engine-persistent        %8.0f MiB", probe.engine_persistent_bytes / kMiB);

    if (shadow.ok) {
        emit("  KV: live pass %d blocks -> plan %d blocks, APPLIED (%.0f -> %.0f MiB)", live_kv_blocks,
             shadow.plan.kv.blocks, static_cast<double>(live_kv_blocks) * per_block / kMiB,
             shadow.plan.kv.bytes / kMiB);
    } else {
        emit("  plan REJECTS this configuration (live pass accepted it with %d KV blocks):",
             live_kv_blocks);
        out += "    ";
        // Indent the multi-line failure report by two spaces.
        for (char c : shadow.failure.report()) {
            out += c;
            if (c == '\n')
                out += "    ";
        }
        out += '\n';
    }

    // Both pool ceilings, in the same unit an operator thinks in: sequences.
    // The two pools are sized by different rules - the recurrent state is a
    // fixed pre-charge of max_batch_size slots, the KV pool takes whatever
    // residual is left - so nothing makes them agree, and when they disagree
    // the state was bought for slots the KV pool cannot serve at full context.
    // Printed in BOTH branches: a rejected plan is exactly when the operator
    // needs to know which of the two ceilings is the binding one.
    const int recurrent_seqs = probe.max_batch_size;
    const int kv_seqs =
        shadow.plan.kv.blocks_per_seq > 0 ? shadow.plan.kv.blocks / shadow.plan.kv.blocks_per_seq : 0;
    // The context at which ALL recurrent slots fit: the pool in tokens, split N
    // ways. Without it the common shape (max_seq_len 131072 on a 4596-block
    // pool: M = 0) reads "KV 0 seqs" and names two knobs with no target for
    // either. 4596 x 16 / 28 = 2626 tokens is the answer the operator needs.
    const int ctx_all = recurrent_seqs > 0
                            ? shadow.plan.kv.blocks * probe.kv_block_size / recurrent_seqs
                            : 0;
    emit("  ceiling: recurrent %d seqs, KV %d seqs at max_seq_len %d (%d blocks/seq), all %d seqs at "
         "<= %d tokens",
         recurrent_seqs, kv_seqs, probe.max_seq_len, shadow.plan.kv.blocks_per_seq, recurrent_seqs,
         ctx_all);
    // Only where there IS per-slot state to over-buy. A dense pool that holds
    // fewer full-context sequences than max_batch_size is ordinary continuous
    // batching: nothing was pre-charged per slot, so nothing was wasted.
    if (probe.ssm_state_bytes > 0 && kv_seqs < recurrent_seqs)
        emit("  WARN: the KV pool serves %d of the %d recurrent slots at max_seq_len %d; all %d fit "
             "only up to %d tokens of context (%d blocks x %d). Set runtime.max_seq_len=%d, or "
             "lower runtime.max_batch_size",
             kv_seqs, recurrent_seqs, probe.max_seq_len, recurrent_seqs, ctx_all,
             shadow.plan.kv.blocks, probe.kv_block_size, ctx_all);

    // Say what is NOT modelled rather than implying full coverage.
    if (!probe.workspace_estimate_available)
        emit("  note: executor workspaces not modelled in this probe");
    if (probe.vision_tower_unmodelled)
        emit("  note: vision tower not in the shadow plan (the arena does reserve it, from a "
             "pre-load probe)");
    emit("  note: forward scratch not modelled until A7 step 4 measures its high-water");
    return out;
}

}  // namespace imp
