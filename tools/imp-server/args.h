#pragma once

#include <string>
#include <utility>
#include <vector>

struct ServerArgs {
    // imp.conf integration. --config overrides the search-path default;
    // --set is a repeatable key=value applied on top.
    std::string config_path;
    std::vector<std::string> config_overrides;

    std::string model_path;
    std::string revision;  // --revision: HuggingFace model revision (branch/tag/commit)
    std::string host = "127.0.0.1";
    int port = 8080;
    int max_tokens = 8192;
    int gpu_layers = -1;  // -1 = all on GPU
    int device = 0;
    std::string chat_template = "auto";
    // --lora NAME=PATH (repeatable): PEFT adapters loaded at startup,
    // selectable per request via the "lora" body field.
    std::vector<std::pair<std::string, std::string>> loras;
    bool no_cuda_graphs = false;
    bool ssm_fp16 = false;
    bool kv_fp8 = false;
    bool kv_int8 = false;
    bool kv_int4 = false;
    bool kv_nvfp4 = false;
    bool kv_mxfp4 = false;
    bool kv_turboquant = false;      // [DEPRECATED] alias for kv_mxfp4
    bool kv_turboquant_lite = false; // [DEPRECATED] alias for kv_mxfp4
    int turboquant_sketch_mult = 2;  // [DEPRECATED] ignored
    int prefill_chunk_size = -1;  // -1 = per-arch default (512 if supported, 0 otherwise)
    int decode_nvfp4 = -1;                      // -1=auto, 0=off, 1=additive, 2=NVFP4-only
    bool mxfp4_prefill = false;                 // --mxfp4-prefill: CUTLASS MXFP4 GEMM for prefill
    bool dual_path_quant = false;               // --dual-path-quant: FP8 attention + NVFP4 FFN
    std::string mmproj_path;                    // --mmproj: vision encoder GGUF
    std::string models_dir;                     // --models-dir: scan for .gguf files
    std::string api_key;                        // --api-key: require Bearer token auth
    std::string reasoning_format = "deepseek";  // --reasoning-format: deepseek or none
    float think_budget =
        0.5f;  // --think-budget: fraction of max_tokens for reasoning (1.0=unlimited, 0=disabled).
               // 0.5 matches docs/usage.md and guarantees answer headroom — at 1.0 a
               // rambling reasoning model eats max_tokens and returns empty content.
    int min_kv_tokens = 0;  // --min-kv-tokens: minimum KV cache capacity (0=auto)
    // Server limits
    int max_concurrent = 64;        // --max-concurrent: max simultaneous requests (0=unlimited)
    int request_timeout = 300;      // --request-timeout: per-request timeout in seconds (0=unlimited)
    int rate_limit = 0;             // --rate-limit: max requests per minute per IP (0=unlimited)
    int max_input_tokens = 0;       // --max-input-tokens: reject prompts longer than this (0=unlimited)
    std::string prefix_cache_path;  // --prefix-cache: path to persist prefix cache
    std::string log_requests_path;  // --log-requests: append JSONL of every chat/messages request
};

ServerArgs parse_server_args(int argc, char** argv);
void print_server_usage(const char* prog);
