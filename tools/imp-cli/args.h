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
    bool interactive = false;
    int device = 0;
    int gpu_layers = -1;       // -1 = all on GPU
    bool ssm_fp16 = false;     // Use FP16 for SSM h_state
    std::string chat_template = "auto";  // auto, none, chatml, llama2, llama3, nemotron
};

CliArgs parse_args(int argc, char** argv);
void print_usage(const char* prog);
