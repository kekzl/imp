#pragma once

// Diagnostics configuration, one of the nine sections split out of
// core/dispatch_policy.h on 2026-08-21.
//
// WHY. dispatch_policy.h aggregates all nine and is included by 23 translation
// units, of which 21 touch two sections or fewer. Adding one field to it costs
// 137.1 s of incremental rebuild, against 9.1 s for a small .cpp and 14.6 s for
// the largest .cu the file-size gate polices. A TU that needs only this section
// can include only this header and stop rebuilding when the others change.
//
// This is F-10 one level down, and dispatch_policy.h's own preamble records the
// original: config.h was included by 22 files, 85 TUs transitively, and changed
// 130 times in six months - "the highest build cost in the repo". Lifting nine
// sections into an aggregate fixed that, and gave the aggregate the same
// property for the same reason.
//
// Pure move: the contents below are byte-identical to their previous form, and
// dispatch_policy.h includes every one of these, so no existing include breaks.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct Diagnostics {
    // Process log level: "debug" | "info" | "warn" | "error" | "fatal".
    // Applied by process_diag_install(), which runs from both tool mains
    // AND Engine::init, so a C-API consumer reaches it too.
    // Until 2026-08-03 there was no way to set this at all: log_set_level()
    // was the only writer of g_log_level and nothing called it, so the level
    // was pinned at INFO and all 76 IMP_LOG_DEBUG sites were unreachable —
    // a debug facility that could not be switched on.
    std::string log_level = "info";
    bool debug_forward = false;
    bool debug_template = false;
    std::string dump_hidden_dir;
    std::string dump_logits_dir;   // path or empty
    std::string dump_routing_dir;  // path or empty
    // Path for the per-layer MoE expert-activation histogram (JSON), written at
    // executor teardown. Empty = off. Unlike dump_routing_dir, which logs one
    // token's top-k as a DEBUG line, this counts EVERY routing decision of the
    // run — the dataset the resident/host expert-split question needs.
    std::string moe_expert_hist;
    // Path for a per-token MoE expert TRACE (JSON), written at executor
    // teardown. Empty = off. The histogram above is an aggregate and cannot see
    // temporal locality, which is the whole question for a cache: an LRU pays
    // only if an expert selected now is selected again soon. Decode only (n==1),
    // so each record is one (token, layer).
    std::string moe_expert_trace;
    bool dump_tokens = false;
    // Teacher-forced perplexity: restrict the NLL sum to logit rows
    // i in [ppl_first, ppl_last] (row i predicts token i+1); ppl_last=-1
    // means "through the end" (n-2). Matches llama-perplexity's window
    // (`first = n_ctx/2`, rows [first, n_ctx-2]) when llama.cpp runs the
    // same corpus with `-c C --chunks 1` (llama-perplexity refuses
    // single-chunk-over-everything: it wants >= 2*n_ctx total tokens):
    // set ppl_first = C/2, ppl_last = C-2 (+ token-offset if the streams
    // are BOS-shifted). The cross-engine PPL-parity bar (GOAL release
    // bar 1) is measured this way.
    int ppl_first = 0;
    int ppl_last = -1;
    int exit_layer = -1;
    bool profile = false;
    bool graph_diag = false;
    // Trace knobs that used to be raw getenv() reads in the hot path
    // (#1207). CLAUDE.md's rule is that IMP_DETERMINISTIC and IMP_FMHA_FA2
    // are the only seeded env vars; IMP_SPEC_TRACE / IMP_JUMP_TRACE /
    // IMP_PPL_DUMP had crept back in. The env names still work — they are
    // debug aids people have in their shell history — but they are seeded
    // into these keys at load, so `--set` and imp.conf reach them too and
    // `imp-cli --help`/imp.conf.example can document them.
    bool spec_trace = false;  // per-step speculative draft/verify trace
    bool jump_trace = false;  // conditional-graph jump trace
    std::string ppl_dump;     // path: dump per-token NLL from --perplexity
    std::string graph_dump_dir;
    // Force NVFP4 dispatch through dequant->FP16 GEMV (M=1 bisection
    // tool — see Mistral-Small-3.2-NVFP4 long-form repetition loops).
    bool nvfp4_force_dequant = false;
    // Skip building the NVFP4 decode cache entirely (bisection/eval
    // tool): decode runs on the source-precision paths (dp4a GEMV for
    // GGUF quants, FP16 GEMV otherwise) — the pre-cache decode
    // semantics. Distinct from nvfp4_force_dequant, which dequantizes
    // the already-NVFP4-quantized cache and so keeps NVFP4 values.
    bool no_nvfp4_decode_cache = false;
    // Do not let the NVFP4 dequant-workspace cap disable prefill-graph
    // capture (`executor_workspace_buffers.cu`, kCap = 512 MiB). Probe
    // tool: the cap is compared against the largest dequant target,
    // which for every model with a big vocabulary is the LM head
    // (vocab x d_model x 2 B — 593 MiB on Qwen3-Coder-30B, 1187 MiB on
    // Qwen3-8B), so prefill capture is off for every model above ~1B.
    // Whether the M>1 NVFP4 dequant fallback can actually be reached
    // under prefill capture is the open question; with this set, the
    // existing fail-loud in gemm_nvfp4 answers it — capture succeeds if
    // the path never fires, and fails cleanly if it does.
    bool prefill_graph_ignore_dequant_cap = false;
    // #847 graph-captured-verify feasibility probe: stream-capture every
    // spec verify chunk forward, instantiate + launch the graph (falls
    // back to the eager forward on any failure). Logs per-attempt
    // outcomes — a capturability census, not a perf path.
    // Capture-fidelity check (diagnostics). When on, every replay of a cached
    // verify-chunk graph is compared against an eager forward of the same state:
    // half A eager, restore the recurrent slab from the pre-chunk copy, half B
    // the cached graph, then diff the row-0 logits. Costs a full extra forward
    // plus two vocab-sized D2H per verify step, so it is a gate/diagnostic mode,
    // never a serving one. Off it is one bool test per verify step.
    bool spec_capture_fidelity = false;
    bool spec_capture_probe = false;
    // Log shape + per-candidate algoId/tileId + chosen algo for every
    // benchmark_and_select_algo call.
    bool log_gemm_algo = false;
    // MTP pattern logging (predicted, actual, match per step).
    bool mtp_pattern_log = false;
    // Stage 0 tree-ceiling probe: ask the MTP head for its top-4 candidates on
    // every chain step and tally, per depth, whether the true next token was
    // within top-w. imp-cli prints the table at the end of a run.
    //
    // Off by default because it is not free, and it was not free in the serving
    // path either: measured on Qwen3.8-27B, the top-4 kernel is a single
    // <<<1,256>>> block scanning a 248 320-entry vocabulary once per width,
    // 713 us per drafted token — twice the cost of the lm_head GEMV that
    // produced the logits, and 12 % of all GPU time in an MTP run. Asking for
    // it also forces a per-draft cudaStreamSynchronize, which is the very
    // thing the device-side chain exists to avoid. Nothing in serving reads
    // the result.
    bool mtp_tree_probe = false;
    // MTP: pass main model's post-RMSNorm hidden to draft head (vLLM
    // variant).
    bool mtp_prenorm_h = false;
    // Audit NVFP4 weight scales at load time.
    bool audit_nvfp4_scales = false;
    // Per-component VRAM accounting harness (MemAccount): lifecycle
    // checkpoints + per-pool notes + device-used peak sampler. Default off
    // (zero overhead). See src/memory/mem_account.h.
    bool vram_audit = false;
    // Optional append-only file the VRAM audit table is mirrored into.
    std::string vram_audit_dump;
    // [RETIRED] tq_skip_qjl removed in Phase 5 (TurboQuant retired 2026-05-17).
};
}  // namespace imp::cfg
