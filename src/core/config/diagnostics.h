#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct Diagnostics {
    // Process log level: debug|info|warn|error|fatal. Applied by
    // process_diag_install(), called from both tool mains and Engine::init, so
    // C-API consumers reach it too.
    std::string log_level = "info";
    bool debug_forward = false;
    bool debug_template = false;
    std::string dump_hidden_dir;
    std::string dump_logits_dir;   // path or empty
    // Directory for the final LM logits per forward pass ([rows,vocab] FP32
    // .npy). Distinct from dump_logits_dir (MoE gate logits at one layer): this
    // is what the sampler sees, written post soft-cap. Empty = off.
    std::string dump_final_logits_dir;
    // Directory for the GDN recurrent state: gdn_state.npy ([n_gdn_layers,
    // heads*head_dim*state_size] FP32, last pass wins) + gdn_state_stats.jsonl
    // (per-pass min/max/RMS/non-finite count). State is FP32 by contract (mamba_ssm_dtype). Empty = off.
    std::string dump_gdn_state_dir;
    std::string dump_routing_dir;  // path or empty
    // Path for the per-layer MoE expert-activation histogram (JSON), written at
    // executor teardown. Counts every routing decision of the run (unlike
    // dump_routing_dir's one-token DEBUG line). Empty = off.
    std::string moe_expert_hist;
    // Path for a per-token MoE expert trace (JSON), written at executor
    // teardown. Unlike the histogram, preserves temporal locality (needed to
    // judge an LRU expert cache). Decode only (n==1): one record per (token, layer).
    std::string moe_expert_trace;
    bool dump_tokens = false;
    // Teacher-forced PPL: sum NLL over rows [ppl_first, ppl_last] (row i
    // predicts token i+1); ppl_last=-1 means through n-2. Matches
    // llama-perplexity's window (first=n_ctx/2, [first, n_ctx-2]) for cross-engine PPL parity (GOAL bar 1).
    int ppl_first = 0;
    int ppl_last = -1;
    int exit_layer = -1;
    bool profile = false;
    bool graph_diag = false;
    // Was a raw getenv() read in the hot path (#1207). IMP_SPEC_TRACE /
    // IMP_JUMP_TRACE / IMP_PPL_DUMP still work as env names but are seeded into
    // these keys at load, so --set and imp.conf reach them too.
    bool spec_trace = false;  // per-step speculative draft/verify trace
    bool jump_trace = false;  // conditional-graph jump trace
    std::string ppl_dump;     // path: dump per-token NLL from --perplexity
    std::string graph_dump_dir;
    // Force NVFP4 dispatch through dequant->FP16 GEMV (M=1 bisection
    // tool — see Mistral-Small-3.2-NVFP4 long-form repetition loops).
    bool nvfp4_force_dequant = false;
    // Skip building the NVFP4 decode cache (bisection/eval): decode runs the
    // pre-cache source-precision paths (dp4a GEMV for GGUF, FP16 GEMV
    // otherwise). Distinct from nvfp4_force_dequant (dequantizes the NVFP4 cache itself).
    bool no_nvfp4_decode_cache = false;
    // Probe: don't let the NVFP4 dequant-workspace cap (kCap=512 MiB,
    // executor_workspace_buffers.cu) disable prefill-graph capture. With this
    // set, gemm_nvfp4's fail-loud path answers whether the M>1 dequant fallback is reached under capture.
    bool prefill_graph_ignore_dequant_cap = false;
    // #847 feasibility probe: stream-captures every verify chunk forward, falls
    // back to eager on failure; logs a capturability census, not a perf path.
    // Fidelity check: replays a cached verify graph vs an eager forward and
    // diffs row-0 logits (costs a full extra forward + 2 D2H per step). Diagnostic only.
    bool spec_capture_fidelity = false;
    bool spec_capture_probe = false;
    // Log shape + per-candidate algoId/tileId + chosen algo for every
    // benchmark_and_select_algo call.
    bool log_gemm_algo = false;
    // MTP pattern logging (predicted, actual, match per step).
    bool mtp_pattern_log = false;
    // Host-side phase attribution for batched decode steps (build / forward /
    // sample / distribute / outside), aggregated every 256 steps. The GPU gap
    // profile locates idle; this says which HOST phase produces it.
    bool step_timing = false;
    // imp-server worker-loop phase attribution (admission/engine step/
    // delivery), aggregated every 256 loops; companion to step_timing
    // (engine-phase vs worker-around-it). Env IMP_WORKER_TIMING seeded into this key at load.
    bool worker_timing = false;
    // Stage 0 tree-ceiling probe: for every chain step, ask the MTP head for
    // top-4 candidates and tally whether the true next token was in top-w per
    // depth. Off by default: costs a full vocab-scan kernel plus a per-draft
    // sync (the very thing the device-side chain avoids). Nothing in serving reads the result.
    bool mtp_tree_probe = false;
    // MTP: feed the draft head the main model's post-RMSNorm hidden (vLLM
    // variant), matching how the head was trained; imp's executor hidden_ is
    // pre-norm, so this normalizes the fed pair. Default on; false restores the pre-norm feed.
    bool mtp_prenorm_h = true;
    // Audit NVFP4 weight scales at load time.
    bool audit_nvfp4_scales = false;
    // Per-component VRAM accounting harness (MemAccount): lifecycle
    // checkpoints + per-pool notes + device-used peak sampler. Default off
    // (zero overhead). See src/memory/mem_account.h.
    bool vram_audit = false;
    // Optional append-only file the VRAM audit table is mirrored into.
    std::string vram_audit_dump;
    // Finished bisect kept as a diagnostic (AUDIT_arch_2026 A2-9): LM head runs
    // dequant->FP16->cuBLAS instead of the fused Q8_1 GEMV, isolating a wrong
    // top logit from a wrong hidden state. Never on in serving.
    bool lm_dequant_fp16 = false;
    // [RETIRED] tq_skip_qjl removed in Phase 5 (TurboQuant retired 2026-05-17).
};
}  // namespace imp::cfg
