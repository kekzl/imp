#pragma once

#include <string>

struct CliArgs {
    std::string model_path;
    std::string prompt;
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    int seed = -1;
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    bool interactive = false;
    int device = 0;
    int gpu_layers = -1;       // -1 = all on GPU
    bool ssm_fp16 = false;     // Use FP16 for SSM h_state
    bool no_cuda_graphs = false;  // Disable CUDA Graph capture for decode
    std::string chat_template = "auto";  // auto, none, chatml, llama2, llama3, nemotron
    int prefill_chunk_size = 0;  // --prefill-chunk-size: 0 = no chunking
    bool bench = false;        // --bench: synthetic benchmark mode
    int bench_pp = 512;        // --bench-pp: synthetic prompt token count
    int bench_reps = 3;        // --bench-reps: repetitions to average
};

CliArgs parse_args(int argc, char** argv);
void print_usage(const char* prog);
