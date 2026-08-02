#pragma once

// The CLI flags imp-cli and imp-server both accept, in one place (#1209).
//
// 26 flags were parsed by two hand-written else-if chains that had to agree by
// review alone — a fix applied to one binary and not the other is invisible
// until someone hits it. Verified before consolidating: every one of the 26
// handlers was byte-identical in both parsers, and every default matched except
// `max_tokens` — 256 in imp-cli, 8192 in imp-server. That gap turned out to be
// age, not intent (see the field), so both now use 8192 and the shared struct
// needs no per-tool override at all.
//
// Structured as a BASE CLASS rather than a member so that `args.model_path`
// keeps working at every existing use site in both binaries — the consolidation
// touches the two parsers and nothing else. Should a per-tool default ever be
// needed again, a derived struct just assigns it in its own initialiser.

#include <string>
#include <vector>

// Flags shared by imp-cli and imp-server. Parsed by parse_common_flag().
struct CommonArgs {
    // imp.conf integration. --config overrides the search-path default; --set is
    // a repeatable key=value (e.g. --set kv_cache.dtype=fp8) applied on top.
    // Unknown keys are an ERROR for --set (a typo silently doing nothing is how
    // three scoring runs once ran without the determinism they claimed).
    std::string config_path;
    std::vector<std::string> config_overrides;

    std::string model_path;
    std::string revision;  // --revision: HuggingFace model revision (branch/tag/commit)

    int device = 0;
    int gpu_layers = -1;  // -1 = all on GPU
    // 8192 for both tools. imp-cli sat at 256 until #1209 — a default from
    // before reasoning models, where the think block alone overruns it and the
    // answer comes back empty with finish_reason=length. On a 32 GB card the
    // output length is not the scarce resource; KV capacity is, and that is
    // sized separately (max_seq_len auto + the planner's min_kv_tokens floor).
    int max_tokens = 8192;

    std::string chat_template = "auto";  // auto, none, chatml, llama2, llama3, nemotron, gemma
    std::string mmproj_path;             // --mmproj: vision encoder GGUF

    bool mem_report = false;      // --mem-report: full VRAM attribution table at init
    int vram_budget_mb = 0;       // --vram-budget: hard per-process VRAM cap in MiB (0 = uncapped)
    int min_kv_tokens = 0;        // --min-kv-tokens: floor KV capacity (0 = auto)
    bool no_cuda_graphs = false;  // disable CUDA Graph capture for decode
    int prefill_chunk_size = -1;  // >=0 = explicit chunk, -1 = per-arch engine default

    // KV cache dtype selection (mutually exclusive in practice; last flag wins).
    bool kv_fp8 = false;    // FP8 E4M3 (half size)
    bool kv_int8 = false;   // INT8 with dp4a attention
    bool kv_int4 = false;   // INT4 (quarter size)
    bool kv_nvfp4 = false;  // NVFP4 (FP4 E2M1 + UE4M3 scales)
    bool kv_mxfp4 = false;  // MXFP4-KV (packed FP4 + UE8M0 scales)

    bool ssm_fp16 = false;         // FP16 for SSM h_state
    int decode_nvfp4 = -1;         // -1=auto, 0=off, 1=additive, 2=NVFP4-only
    bool mxfp4_prefill = false;    // --mxfp4-prefill: CUTLASS MXFP4 GEMM for prefill
    bool dual_path_quant = false;  // --dual-path-quant: FP8 attention + NVFP4 FFN
};

// Consumes argv[i] if it is one of the shared flags, advancing `i` past any
// value it takes, and returns true. Returns false (leaving `i` untouched) when
// the flag is not one of ours, so the caller falls through to its own chain.
//
// Callers must invoke this BEFORE their tool-specific chain: the two binaries
// must not be able to shadow a shared flag with a divergent local handler,
// which is the failure this consolidation exists to prevent.
bool parse_common_flag(CommonArgs& args, int argc, char** argv, int& i);
