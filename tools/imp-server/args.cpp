#include "args.h"
#include "common/args_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

void print_server_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Options:\n"
            "  --model <path>        Path to model file or HuggingFace repo ID (optional)\n"
            "  --revision <rev>      HuggingFace model revision (branch, tag, or commit hash)\n"
            "  --config <path>       imp.conf path (default: ./imp.conf, ~/.config/imp/imp.conf)\n"
            "  --set <sec.key=val>   Override one imp.conf key (repeatable)\n"
            "  --host <addr>         Listen address (default: 127.0.0.1)\n"
            "  --port <n>            Listen port (default: 8080)\n"
            "  --max-tokens <n>      Default max tokens (default: 8192)\n"
            "  --max-batch <n>       Decode batch / KV+workspace sizing (default: 0 = auto)\n"
            "  --gpu-layers <n>      Layers on GPU, -1 = all (default: -1)\n"
            "  --device <n>          CUDA device ID (default: 0)\n"
            "  --chat-template <t>   auto, none, chatml, llama2, llama3, nemotron, gemma\n"
            "  --lora NAME=PATH      Load a PEFT LoRA adapter (repeatable); select per request via \"lora\" "
            "field\n"
            "  --no-cuda-graphs      Disable CUDA Graph capture for decode\n"
            "  --ssm-fp16            Use FP16 for SSM h_state\n"
            "  --kv-fp8              Use FP8 E4M3 KV cache (halves KV memory)\n"
            "  --kv-int8             Use INT8 KV cache with dp4a attention\n"
            "  --kv-int4             Use INT4 KV cache (quarters KV memory)\n"
            "  --kv-nvfp4            Use NVFP4 KV cache (quarters KV memory; FP4 + E4M3 scales)\n"
            "  --kv-mxfp4            Use MXFP4-KV cache (quarters KV memory; FP4 + UE8M0 scales)\n"
            "  --prefill-chunk-size <n> Max tokens per prefill chunk (0 = single-chunk, default: per-arch)\n"
            "  --min-kv-tokens <n>   Minimum KV capacity in tokens (default: auto)\n"
            "  --prefix-cache <path> Persist the prefix cache to this path across restarts\n"
            "  --decode-nvfp4        NVFP4 decode cache (additive: FP16 prefill + NVFP4 decode)\n"
            "  --decode-nvfp4-only   NVFP4 decode cache (replacement: saves VRAM, slower prefill)\n"
            "  --mxfp4-prefill       Use CUTLASS MXFP4 GEMM for prefill (sm_120, requires NVFP4)\n"
            "  --dual-path-quant     FP8 attention + NVFP4 FFN (higher quality attention, faster FFN)\n"
            "  --no-nvfp4            Disable NVFP4 decode cache (override auto-detection)\n"
            "  --mmproj <path>       Path to vision encoder GGUF (mmproj) for multimodal\n"
            "  --models-dir <path>   Directory to scan for .gguf models (auto-load on select)\n"
            "  --api-key <key>       Require Bearer token authentication\n"
            "  --metrics-require-auth  Also gate /metrics behind --api-key\n"
            "  --allow-remote-images Fetch http(s) image_url from request bodies (default: off).\n"
            "                        Off, the server never connects anywhere a caller names.\n"
            "  --reasoning-format <f> deepseek (default) or none\n"
            "  --mem-report          Print the full VRAM attribution table at init\n"
            "  --vram-budget <mb>    Hard per-process VRAM cap in MiB — size everything as if\n"
            "                        the GPU only had this much (multi-server on one GPU)\n"
            "  --think-budget <f>    Fraction of max_tokens for reasoning (default: 0.5, 0=disabled)\n"
            "  --max-concurrent <n>  Max simultaneous requests (default: 64, 0=unlimited)\n"
            "  --request-timeout <s> Per-request timeout in seconds (default: 300, 0=unlimited)\n"
            "  --rate-limit <n>      Max requests/minute per IP (default: 0=unlimited)\n"
            "  --max-input-tokens <n> Reject prompts longer than n tokens with HTTP 400 (0=unlimited)\n"
            "  --log-requests <path> Append per-request JSONL with prompt + response content\n"
            "  --help                Show this help message\n",
            prog);
}

ServerArgs parse_server_args(int argc, char** argv) {
    ServerArgs args;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        // Shared imp-cli/imp-server flags first (#1209): a tool must not be able
        // to shadow one with a divergent local handler.
        if (parse_common_flag(args, argc, argv, i))
            continue;

        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_server_usage(argv[0]);
            std::exit(0);
        } else if (std::strcmp(arg, "--host") == 0 && i + 1 < argc) {
            args.host = argv[++i];
        } else if (std::strcmp(arg, "--port") == 0 && i + 1 < argc) {
            args.port = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--max-batch") == 0 && i + 1 < argc) {
            args.max_batch_size = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--lora") == 0 && i + 1 < argc) {
            std::string spec = argv[++i];
            size_t eq = spec.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 >= spec.size()) {
                fprintf(stderr, "--lora expects NAME=PATH, got '%s'\n", spec.c_str());
                exit(1);
            }
            args.loras.emplace_back(spec.substr(0, eq), spec.substr(eq + 1));
        } else if (std::strcmp(arg, "--models-dir") == 0 && i + 1 < argc) {
            args.models_dir = argv[++i];
        } else if (std::strcmp(arg, "--api-key") == 0 && i + 1 < argc) {
            args.api_key = argv[++i];
        } else if (std::strcmp(arg, "--metrics-require-auth") == 0) {
            args.metrics_require_auth = true;
        } else if (std::strcmp(arg, "--allow-remote-images") == 0) {
            args.allow_remote_images = true;
        } else if (std::strcmp(arg, "--reasoning-format") == 0 && i + 1 < argc) {
            args.reasoning_format = argv[++i];
        } else if (std::strcmp(arg, "--think-budget") == 0 && i + 1 < argc) {
            args.think_budget = std::atof(argv[++i]);
        } else if (std::strcmp(arg, "--max-concurrent") == 0 && i + 1 < argc) {
            args.max_concurrent = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--request-timeout") == 0 && i + 1 < argc) {
            args.request_timeout = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--rate-limit") == 0 && i + 1 < argc) {
            args.rate_limit = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--max-input-tokens") == 0 && i + 1 < argc) {
            args.max_input_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--prefix-cache") == 0 && i + 1 < argc) {
            args.prefix_cache_path = argv[++i];
        } else if (std::strcmp(arg, "--log-requests") == 0 && i + 1 < argc) {
            args.log_requests_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            print_server_usage(argv[0]);
            std::exit(1);
        }
    }

    return args;
}
