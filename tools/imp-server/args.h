#pragma once

#include <string>

struct ServerArgs {
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    int max_tokens = 8192;
    int gpu_layers = -1;       // -1 = all on GPU
    int device = 0;
    std::string chat_template = "auto";
    bool no_cuda_graphs = false;
    bool ssm_fp16 = false;
    bool kv_fp8 = false;
    bool kv_int8 = false;
    int prefill_chunk_size = 0;
    int decode_nvfp4 = -1;     // -1=auto, 0=off, 1=additive, 2=NVFP4-only
    std::string mmproj_path;   // --mmproj: vision encoder GGUF
    std::string models_dir;    // --models-dir: scan for .gguf files
    std::string api_key;       // --api-key: require Bearer token auth
    std::string reasoning_format = "deepseek";  // --reasoning-format: deepseek or none
    float think_budget = 0.5f; // --think-budget: fraction of max_tokens for reasoning (0=disabled)
    std::string preset;        // --preset: named model preset (e.g. qwen3-32b)
    std::string presets_file;  // --presets-file: custom presets.toml path
};

ServerArgs parse_server_args(int argc, char** argv);
void print_server_usage(const char* prog);
