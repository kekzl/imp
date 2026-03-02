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
    int seed = -1;
    float min_p = 0.0f;
    float typical_p = 1.0f;        // Locally typical sampling (1.0 = disabled)
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
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
    int decode_nvfp4 = 0;      // --decode-nvfp4: 1=additive, --decode-nvfp4-only: 2=replacement
    std::vector<std::string> stop_sequences;  // --stop: text-level stop strings
    bool bench = false;        // --bench: synthetic benchmark mode
    int bench_pp = 512;        // --bench-pp: synthetic prompt token count
    int bench_reps = 3;        // --bench-reps: repetitions to average
};

CliArgs parse_args(int argc, char** argv);
void print_usage(const char* prog);
