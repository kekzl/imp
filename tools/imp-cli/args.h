#pragma once

#include "common/args_common.h"

#include <string>
#include <vector>

struct CliArgs : CommonArgs {
    // Shared flags live in CommonArgs (#1209). Inherited, not composed, so
    // every existing `args.<field>` use site is unchanged.

    // imp.conf integration. --config overrides the search-path default;
    // --set is a repeatable key=value (e.g. --set kv_cache.dtype=fp8) that
    // applies on top of the loaded config. See imp.conf.example for keys.

    std::string prompt;
    std::string perplexity_file;  // --perplexity <file>: teacher-forced PPL over the file's text
    // --calibrate <out>: collect per-channel activation magnitudes during the
    // --perplexity pass and write them for imp-quantize --calib.
    std::string calibrate_out;
    int max_seq_len = 0;    // --max-seq-len: KV context ceiling (0 = auto from VRAM)
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
    int mtp_spec_decode_k = 0;           // --mtp-spec-decode K: 0 = off, >0 = MTP draft length
    bool prefill_fp8 = false;            // --prefill-fp8: use FP8 E4M3 weight cache for prefill
    bool prefix_caching = false;         // --prefix-caching: reuse KV blocks for shared prefixes
    bool streaming_kv = false;           // --streaming-kv: StreamingLLM smart KV cache (sinks + window)
    bool no_streaming_kv_auto = false;   // --no-streaming-kv-auto: disable auto-StreamingLLM on KV pressure
    int streaming_sinks = 4;             // --stream-sinks: # of sink tokens to keep
    int streaming_window = 0;            // --stream-window: window size (0 = use ModelConfig::sliding_window)
    std::vector<std::string> stop_sequences;  // --stop: text-level stop strings
    bool bench = false;                       // --bench: synthetic benchmark mode
    int bench_pp = 512;                       // --bench-pp: synthetic prompt token count
    int bench_reps = 3;                       // --bench-reps: repetitions to average
    // --image, repeatable: images for vision, in the order given. Each one
    // gets its own placeholder in the prompt.
    std::vector<std::string> image_paths;
};

CliArgs parse_args(int argc, char** argv);
void print_usage(const char* prog);
