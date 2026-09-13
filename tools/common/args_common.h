#pragma once

// CLI flags imp-cli and imp-server both accept, in one place (#1209): two hand-written
// else-if chains had to agree by review alone, so a fix to one binary was invisible in the other.
// Structured as a BASE CLASS (not a member) so `args.model_path` keeps working at every
// existing use site; a per-tool default can still be added via a derived struct's initialiser.

#include <string>
#include <vector>

// Flags shared by imp-cli and imp-server. Parsed by parse_common_flag().
struct CommonArgs {
    // imp.conf integration: --config overrides the search-path default; --set is a repeatable
    // key=value (e.g. --set kv_cache.dtype=fp8) applied on top. Unknown keys are an ERROR for
    // --set: a silent no-op once let scoring runs claim determinism they didn't have.
    std::string config_path;
    std::vector<std::string> config_overrides;

    std::string model_path;
    std::string revision;  // --revision: HuggingFace model revision (branch/tag/commit)

    int device = 0;
    int gpu_layers = -1;  // -1 = all on GPU
    // 8192 for both tools: imp-cli's old default of 256 predates reasoning models, where the
    // think block alone overruns it and the answer returns empty (finish_reason=length).
    // Output length is not the scarce resource on a 32GB card; KV capacity is, sized separately
    // (max_seq_len auto + the planner's min_kv_tokens floor).
    int max_tokens = 8192;

    std::string chat_template = "auto";  // auto, none, chatml, llama2, llama3, nemotron, gemma
    std::string mmproj_path;             // --mmproj: vision encoder GGUF

    // --json: stdout carries EXACTLY one JSON document and nothing else, so a
    // caller can pipe it into jq without a regex over a column layout that is
    // not a contract (#1583). Everything human-readable goes to stderr.
    bool json_out = false;

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

// Consumes argv[i] if it's a shared flag (advancing i past any value), returns true;
// returns false, leaving i untouched, otherwise. Callers must invoke this BEFORE their
// tool-specific chain so a shared flag cannot be shadowed by a divergent local handler.
bool parse_common_flag(CommonArgs& args, int argc, char** argv, int& i);

// imp_calibration_write() takes the path as an argument; --calibrate <out> passed it straight
// through and NOTHING ever read [calibration] out_path, despite it being documented and
// parsed. Resolution order: --calibrate <path> flag, then [calibration] out_path, then empty
// (no calibration run asked for).
std::string resolve_calibration_out(const std::string& flag_value, const std::string& config_value);
