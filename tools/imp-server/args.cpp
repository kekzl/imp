#include "args.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

void print_server_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>        Path to model file (optional; load later via API)\n"
        "  --host <addr>         Listen address (default: 127.0.0.1)\n"
        "  --port <n>            Listen port (default: 8080)\n"
        "  --max-tokens <n>      Default max tokens (default: 2048)\n"
        "  --gpu-layers <n>      Layers on GPU, -1 = all (default: -1)\n"
        "  --device <n>          CUDA device ID (default: 0)\n"
        "  --chat-template <t>   auto, none, chatml, llama2, llama3, nemotron, gemma\n"
        "  --no-cuda-graphs      Disable CUDA Graph capture for decode\n"
        "  --ssm-fp16            Use FP16 for SSM h_state\n"
        "  --kv-fp8              Use FP8 E4M3 KV cache (halves KV memory)\n"
        "  --kv-int8             Use INT8 KV cache with dp4a attention\n"
        "  --prefill-chunk-size <n> Max tokens per prefill chunk (0 = no chunking)\n"
        "  --decode-nvfp4        NVFP4 decode cache (additive: FP16 prefill + NVFP4 decode)\n"
        "  --decode-nvfp4-only   NVFP4 decode cache (replacement: saves VRAM, slower prefill)\n"
        "  --no-nvfp4            Disable NVFP4 decode cache (override auto-detection)\n"
        "  --mmproj <path>       Path to vision encoder GGUF (mmproj) for multimodal\n"
        "  --models-dir <path>   Directory to scan for .gguf models (auto-load on select)\n"
        "  --api-key <key>       Require Bearer token authentication\n"
        "  --help                Show this help message\n",
        prog);
}

ServerArgs parse_server_args(int argc, char** argv) {
    ServerArgs args;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_server_usage(argv[0]);
            std::exit(0);
        } else if (std::strcmp(arg, "--model") == 0 && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (std::strcmp(arg, "--host") == 0 && i + 1 < argc) {
            args.host = argv[++i];
        } else if (std::strcmp(arg, "--port") == 0 && i + 1 < argc) {
            args.port = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--max-tokens") == 0 && i + 1 < argc) {
            args.max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--gpu-layers") == 0 && i + 1 < argc) {
            args.gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--device") == 0 && i + 1 < argc) {
            args.device = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--chat-template") == 0 && i + 1 < argc) {
            args.chat_template = argv[++i];
        } else if (std::strcmp(arg, "--no-cuda-graphs") == 0) {
            args.no_cuda_graphs = true;
        } else if (std::strcmp(arg, "--ssm-fp16") == 0) {
            args.ssm_fp16 = true;
        } else if (std::strcmp(arg, "--kv-fp8") == 0) {
            args.kv_fp8 = true;
        } else if (std::strcmp(arg, "--kv-int8") == 0) {
            args.kv_int8 = true;
        } else if (std::strcmp(arg, "--prefill-chunk-size") == 0 && i + 1 < argc) {
            args.prefill_chunk_size = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--decode-nvfp4") == 0) {
            args.decode_nvfp4 = 1;
        } else if (std::strcmp(arg, "--decode-nvfp4-only") == 0) {
            args.decode_nvfp4 = 2;
        } else if (std::strcmp(arg, "--no-nvfp4") == 0) {
            args.decode_nvfp4 = 0;
        } else if (std::strcmp(arg, "--mmproj") == 0 && i + 1 < argc) {
            args.mmproj_path = argv[++i];
        } else if (std::strcmp(arg, "--models-dir") == 0 && i + 1 < argc) {
            args.models_dir = argv[++i];
        } else if (std::strcmp(arg, "--api-key") == 0 && i + 1 < argc) {
            args.api_key = argv[++i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            print_server_usage(argv[0]);
            std::exit(1);
        }
    }

    return args;
}
