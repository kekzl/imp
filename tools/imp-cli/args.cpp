#include "args.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>        Path to model file (required)\n"
        "  --prompt <text>       Input prompt for generation\n"
        "  --max-tokens <n>      Maximum tokens to generate (default: 256)\n"
        "  --temperature <f>     Sampling temperature (default: 0.7)\n"
        "  --top-p <f>           Top-p (nucleus) sampling (default: 0.9)\n"
        "  --top-k <n>           Top-k sampling (default: 40)\n"
        "  --seed <n>            Random seed, -1 for random (default: -1)\n"
        "  --min-p <f>           Min-p sampling threshold (default: 0.0 = disabled)\n"
        "  --typical-p <f>       Locally typical sampling (default: 1.0 = disabled)\n"
        "  --repeat-penalty <f>  Repetition penalty (default: 1.0 = disabled)\n"
        "  --frequency-penalty <f> Frequency penalty (default: 0.0)\n"
        "  --presence-penalty <f>  Presence penalty (default: 0.0)\n"
        "  --dry-multiplier <f>  DRY n-gram penalty scale (default: 0.0 = disabled)\n"
        "  --dry-base <f>        DRY exponential base (default: 1.75)\n"
        "  --dry-allowed-length <n> DRY: n-grams at or below this not penalized (default: 2)\n"
        "  --dry-penalty-last-n <n> DRY: how far back to scan (default: 0 = all)\n"
        "  --mirostat <n>        Mirostat sampling (0=off, 2=v2) (default: 0)\n"
        "  --mirostat-tau <f>    Mirostat target entropy (default: 5.0)\n"
        "  --mirostat-eta <f>    Mirostat learning rate (default: 0.1)\n"
        "  --interactive         Run in interactive chat mode\n"
        "  --device <n>          CUDA device ID (default: 0)\n"
        "  --gpu-layers <n>      Layers to keep on GPU (-1 = all) (default: -1)\n"
        "  --kv-fp8              Use FP8 E4M3 KV cache (halves KV memory)\n"
        "  --kv-int8             Use INT8 KV cache with dp4a attention (halves KV memory)\n"
        "  --kv-int4             Use INT4 KV cache (quarters KV memory)\n"
        "  --ssm-fp16            Use FP16 for SSM h_state (saves ~50%% SSM VRAM)\n"
        "  --cuda-graphs         (default, no-op — graphs enabled by default)\n"
        "  --no-cuda-graphs      Disable CUDA Graph capture for decode\n"
        "  --chat-template <t>   Chat template: auto, none, chatml, llama2, llama3, nemotron, gemma, deepseek_r1, phi\n"
        "  --prefill-chunk-size <n> Max tokens per prefill chunk (default: 0 = no chunking)\n"
        "  --prefill-fp8         Use FP8 E4M3 weight cache for ~2x prefill throughput\n"
        "  --decode-nvfp4        NVFP4 decode cache (additive: FP16 prefill + NVFP4 decode)\n"
        "  --decode-nvfp4-only   NVFP4 decode cache (replacement: saves VRAM, slower prefill)\n"
        "  --prefix-caching      Reuse KV cache blocks for shared token prefixes\n"
        "  --mxfp4-prefill       Use CUTLASS MXFP4 GEMM for prefill (sm_120, requires NVFP4)\n"
        "  --no-nvfp4            Disable NVFP4 decode cache (override auto-detection)\n"
        "  --stop <str>          Stop sequence (can specify multiple times, max 4)\n"
        "  --bench               Synthetic benchmark mode (like llama-bench)\n"
        "  --bench-pp <n>        Synthetic prompt token count (default: 512)\n"
        "  --bench-reps <n>      Repetitions to average (default: 3)\n"
        "  --mmproj <path>       Path to vision encoder GGUF (mmproj) for multimodal\n"
        "  --image <path>        Input image for vision (requires --mmproj)\n"
        "  --self-speculative    Self-speculative decoding (layer-skip draft, same model)\n"
        "  --self-spec-k <n>     Draft tokens per self-spec step (default: 2)\n"
        "  --self-spec-exit-layer <n>  Layers to run in draft (-1 = auto)\n"
        "  --self-spec-skip-n <n>  Layers to skip in draft (-1 = auto)\n"
        "  --preset <name|none>  Override auto-detected preset, or 'none' to disable\n"
        "                        Use --preset list to show all available presets\n"
        "  --presets-file <path> Custom presets.toml path\n"
        "  --help                Show this help message\n",
        prog);
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs args;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else if (std::strcmp(arg, "--model") == 0 && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (std::strcmp(arg, "--prompt") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (std::strcmp(arg, "--max-tokens") == 0 && i + 1 < argc) {
            args.max_tokens = std::atoi(argv[++i]);
            args.max_tokens_set = true;
        } else if (std::strcmp(arg, "--temperature") == 0 && i + 1 < argc) {
            args.temperature = static_cast<float>(std::atof(argv[++i]));
            args.temperature_set = true;
        } else if (std::strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
            args.top_p = static_cast<float>(std::atof(argv[++i]));
            args.top_p_set = true;
        } else if (std::strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            args.top_k = std::atoi(argv[++i]);
            args.top_k_set = true;
        } else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            args.seed = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--min-p") == 0 && i + 1 < argc) {
            args.min_p = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--typical-p") == 0 && i + 1 < argc) {
            args.typical_p = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--repeat-penalty") == 0 && i + 1 < argc) {
            args.repetition_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--frequency-penalty") == 0 && i + 1 < argc) {
            args.frequency_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--presence-penalty") == 0 && i + 1 < argc) {
            args.presence_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--repeat-last-n") == 0 && i + 1 < argc) {
            args.repeat_last_n = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--dry-multiplier") == 0 && i + 1 < argc) {
            args.dry_multiplier = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--dry-base") == 0 && i + 1 < argc) {
            args.dry_base = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--dry-allowed-length") == 0 && i + 1 < argc) {
            args.dry_allowed_length = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--dry-penalty-last-n") == 0 && i + 1 < argc) {
            args.dry_penalty_last_n = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--mirostat") == 0 && i + 1 < argc) {
            args.mirostat = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--mirostat-tau") == 0 && i + 1 < argc) {
            args.mirostat_tau = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--mirostat-eta") == 0 && i + 1 < argc) {
            args.mirostat_eta = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--interactive") == 0) {
            args.interactive = true;
        } else if (std::strcmp(arg, "--device") == 0 && i + 1 < argc) {
            args.device = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--gpu-layers") == 0 && i + 1 < argc) {
            args.gpu_layers = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--kv-fp8") == 0) {
            args.kv_fp8 = true;
        } else if (std::strcmp(arg, "--kv-int8") == 0) {
            args.kv_int8 = true;
        } else if (std::strcmp(arg, "--kv-int4") == 0) {
            args.kv_int4 = true;
        } else if (std::strcmp(arg, "--ssm-fp16") == 0) {
            args.ssm_fp16 = true;
        } else if (std::strcmp(arg, "--cuda-graphs") == 0) {
            // no-op: cuda graphs enabled by default now
        } else if (std::strcmp(arg, "--no-cuda-graphs") == 0) {
            args.no_cuda_graphs = true;
        } else if (std::strcmp(arg, "--chat-template") == 0 && i + 1 < argc) {
            args.chat_template = argv[++i];
        } else if (std::strcmp(arg, "--prefill-chunk-size") == 0 && i + 1 < argc) {
            args.prefill_chunk_size = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--prefill-fp8") == 0) {
            args.prefill_fp8 = true;
        } else if (std::strcmp(arg, "--decode-nvfp4") == 0) {
            args.decode_nvfp4 = 1;
        } else if (std::strcmp(arg, "--decode-nvfp4-only") == 0) {
            args.decode_nvfp4 = 2;
        } else if (std::strcmp(arg, "--prefix-caching") == 0) {
            args.prefix_caching = true;
        } else if (std::strcmp(arg, "--mxfp4-prefill") == 0) {
            args.mxfp4_prefill = true;
        } else if (std::strcmp(arg, "--no-nvfp4") == 0) {
            args.decode_nvfp4 = 0;
        } else if (std::strcmp(arg, "--stop") == 0 && i + 1 < argc) {
            if (args.stop_sequences.size() < 4)
                args.stop_sequences.push_back(argv[++i]);
            else
                ++i;  // skip value if at limit
        } else if (std::strcmp(arg, "--bench") == 0) {
            args.bench = true;
        } else if (std::strcmp(arg, "--bench-pp") == 0 && i + 1 < argc) {
            args.bench_pp = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--bench-reps") == 0 && i + 1 < argc) {
            args.bench_reps = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--mmproj") == 0 && i + 1 < argc) {
            args.mmproj_path = argv[++i];
        } else if (std::strcmp(arg, "--image") == 0 && i + 1 < argc) {
            args.image_path = argv[++i];
        } else if (std::strcmp(arg, "--self-speculative") == 0) {
            args.self_speculative = true;
        } else if (std::strcmp(arg, "--self-spec-k") == 0 && i + 1 < argc) {
            args.self_spec_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--self-spec-exit-layer") == 0 && i + 1 < argc) {
            args.self_spec_exit_layer = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--self-spec-skip-n") == 0 && i + 1 < argc) {
            args.self_spec_skip_n = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--preset") == 0 && i + 1 < argc) {
            args.preset = argv[++i];
        } else if (std::strcmp(arg, "--presets-file") == 0 && i + 1 < argc) {
            args.presets_file = argv[++i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    return args;
}
