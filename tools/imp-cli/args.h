#pragma once

#include <string>
#include <vector>

struct CliArgs {
    std::string model_path;
    std::string prompt;
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;

    // Track which values were explicitly set (vs defaults / preset)
    bool max_tokens_set = false;
    bool temperature_set = false;
    bool top_p_set = false;
    bool top_k_set = false;
    int seed = -1;
    float min_p = 0.0f;
    float typical_p = 1.0f;        // Locally typical sampling (1.0 = disabled)
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int repeat_last_n = 0;          // Penalty window (0 = all tokens)
    float dry_multiplier = 0.0f;   // DRY penalty (0=disabled)
    float dry_base = 1.75f;        // DRY exponential base
    int dry_allowed_length = 2;    // N-grams ≤ this not penalized
    int dry_penalty_last_n = 0;    // How far back to scan (0=all)
    int mirostat = 0;              // 0=off, 2=Mirostat v2
    float mirostat_tau = 5.0f;     // Target entropy
    float mirostat_eta = 0.1f;     // Learning rate
    bool interactive = false;
    int device = 0;
    int gpu_layers = -1;       // -1 = all on GPU
    bool kv_fp8 = false;       // Use FP8 E4M3 KV cache (half size)
    bool kv_int8 = false;      // Use INT8 KV cache with dp4a attention
    bool ssm_fp16 = false;     // Use FP16 for SSM h_state
    bool no_cuda_graphs = false;  // Disable CUDA Graph capture for decode
    std::string chat_template = "auto";  // auto, none, chatml, llama2, llama3, nemotron, gemma
    int prefill_chunk_size = 0;  // --prefill-chunk-size: 0 = no chunking
    bool prefill_fp8 = false;  // --prefill-fp8: use FP8 E4M3 weight cache for prefill
    int decode_nvfp4 = -1;     // -1=auto, 0=off, 1=additive, 2=NVFP4-only
    bool mxfp4_prefill = false;  // --mxfp4-prefill: CUTLASS MXFP4 GEMM for prefill
    bool prefix_caching = false;  // --prefix-caching: reuse KV blocks for shared prefixes
    std::vector<std::string> stop_sequences;  // --stop: text-level stop strings
    bool bench = false;        // --bench: synthetic benchmark mode
    int bench_pp = 512;        // --bench-pp: synthetic prompt token count
    int bench_reps = 3;        // --bench-reps: repetitions to average
    std::string mmproj_path;   // --mmproj: vision encoder GGUF
    std::string image_path;    // --image: input image for vision
    bool self_speculative = false;  // --self-speculative: early-exit draft decode
    int self_spec_k = 4;            // --self-spec-k: draft tokens per step
    int self_spec_exit_layer = -1;  // --self-spec-exit-layer: -1 = n_layers/2
    std::string preset;        // --preset: named model preset (e.g. qwen3-32b)
    std::string presets_file;  // --presets-file: custom presets.toml path
};

CliArgs parse_args(int argc, char** argv);
void print_usage(const char* prog);
