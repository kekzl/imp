#pragma once

#include <string>

struct ServerArgs {
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    int max_tokens = 2048;
    int gpu_layers = -1;       // -1 = all on GPU
    int device = 0;
    std::string chat_template = "auto";
    bool no_cuda_graphs = false;
    bool ssm_fp16 = false;
    bool kv_fp8 = false;
    bool kv_int8 = false;
    int prefill_chunk_size = 0;
};

ServerArgs parse_server_args(int argc, char** argv);
void print_server_usage(const char* prog);
