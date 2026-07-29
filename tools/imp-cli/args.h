#pragma once

#include <string>
#include <vector>

struct CliArgs {
    // imp.conf integration. --config overrides the search-path default;
    // --set is a repeatable key=value (e.g. --set kv_cache.dtype=fp8) that
    // applies on top of the loaded config. See imp.conf.example for keys.
    std::string config_path;
    std::vector<std::string> config_overrides;

    std::string model_path;
    std::string revision;  // --revision: HuggingFace model revision (branch/tag/commit)
    std::string prompt;
    std::string perplexity_file;  // --perplexity <file>: teacher-forced PPL over the file's text
    int max_tokens = 256;
    int max_seq_len = 0;    // --max-seq-len: KV context ceiling (0 = auto from VRAM)
    bool mem_report = false; // --mem-report: full VRAM attribution table at init
    int vram_budget_mb = 0; // --vram-budget: hard per-process VRAM cap in MiB (0 = uncapped)
    int min_kv_tokens = 0;  // --min-kv-tokens: floor KV capacity (0 = auto)
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;

    // Track which values were explicitly set (vs defaults / preset)
    bool max_tokens_set = false;
    bool temperature_set = false;
    bool top_p_set = false;
    bool top_k_set = false;
    bool repetition_penalty_set = false;
    int seed = -1;
    float min_p = 0.0f;
    float typical_p = 1.0f;  // Locally typical sampling (1.0 = disabled)
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int repeat_last_n = 0;        // Penalty window (0 = all tokens)
    float dry_multiplier = 0.0f;  // DRY penalty (0=disabled)
    float dry_base = 1.75f;       // DRY exponential base
    int dry_allowed_length = 2;   // N-grams ≤ this not penalized
    int dry_penalty_last_n = 0;   // How far back to scan (0=all)
    int mirostat = 0;             // 0=off, 2=Mirostat v2
    float mirostat_tau = 5.0f;    // Target entropy
    float mirostat_eta = 0.1f;    // Learning rate
    bool interactive = false;
    int device = 0;
    int gpu_layers = -1;                 // -1 = all on GPU
    bool kv_fp8 = false;                 // Use FP8 E4M3 KV cache (half size)
    bool kv_int8 = false;                // Use INT8 KV cache with dp4a attention
    bool kv_int4 = false;                // Use INT4 KV cache (quarter size)
    bool kv_nvfp4 = false;               // Use NVFP4 KV cache (quarter size, FP4 E2M1 + UE4M3 scales)
    bool kv_mxfp4 = false;              // Use MXFP4-KV cache (packed FP4 + UE8M0 scales)
    bool ssm_fp16 = false;               // Use FP16 for SSM h_state
    bool no_cuda_graphs = false;         // Disable CUDA Graph capture for decode
    std::string chat_template = "auto";  // auto, none, chatml, llama2, llama3, nemotron, gemma
    int prefill_chunk_size = -1;         // --prefill-chunk-size: >=0 = explicit chunk, -1 = use engine default (per-arch)
    int mtp_spec_decode_k = 0;           // --mtp-spec-decode K: 0 = off, >0 = MTP draft length
    bool prefill_fp8 = false;            // --prefill-fp8: use FP8 E4M3 weight cache for prefill
    int decode_nvfp4 = -1;               // -1=auto, 0=off, 1=additive, 2=NVFP4-only
    bool mxfp4_prefill = false;          // --mxfp4-prefill: CUTLASS MXFP4 GEMM for prefill
    bool dual_path_quant = false;        // --dual-path-quant: FP8 attention + NVFP4 FFN
    bool prefix_caching = false;         // --prefix-caching: reuse KV blocks for shared prefixes
    bool streaming_kv = false;           // --streaming-kv: StreamingLLM smart KV cache (sinks + window)
    bool no_streaming_kv_auto = false;   // --no-streaming-kv-auto: disable auto-StreamingLLM on KV pressure
    int streaming_sinks = 4;             // --stream-sinks: # of sink tokens to keep
    int streaming_window = 0;            // --stream-window: window size (0 = use ModelConfig::sliding_window)
    std::vector<std::string> stop_sequences;  // --stop: text-level stop strings
    bool bench = false;                       // --bench: synthetic benchmark mode
    int bench_pp = 512;                       // --bench-pp: synthetic prompt token count
    int bench_reps = 3;                       // --bench-reps: repetitions to average
    std::string mmproj_path;                  // --mmproj: vision encoder GGUF
    std::string image_path;                   // --image: input image for vision
};

CliArgs parse_args(int argc, char** argv);
void print_usage(const char* prog);
