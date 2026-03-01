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
        "  --repeat-penalty <f>  Repetition penalty (default: 1.0 = disabled)\n"
        "  --frequency-penalty <f> Frequency penalty (default: 0.0)\n"
        "  --presence-penalty <f>  Presence penalty (default: 0.0)\n"
        "  --interactive         Run in interactive chat mode\n"
        "  --device <n>          CUDA device ID (default: 0)\n"
        "  --gpu-layers <n>      Layers to keep on GPU (-1 = all) (default: -1)\n"
        "  --ssm-fp16            Use FP16 for SSM h_state (saves ~50%% SSM VRAM)\n"
        "  --cuda-graphs         (default, no-op — graphs enabled by default)\n"
        "  --no-cuda-graphs      Disable CUDA Graph capture for decode\n"
        "  --chat-template <t>   Chat template: auto, none, chatml, llama2, llama3, nemotron\n"
        "  --prefill-chunk-size <n> Max tokens per prefill chunk (default: 0 = no chunking)\n"
        "  --bench               Synthetic benchmark mode (like llama-bench)\n"
        "  --bench-pp <n>        Synthetic prompt token count (default: 512)\n"
        "  --bench-reps <n>      Repetitions to average (default: 3)\n"
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
        } else if (std::strcmp(arg, "--temperature") == 0 && i + 1 < argc) {
            args.temperature = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
            args.top_p = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            args.top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            args.seed = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--min-p") == 0 && i + 1 < argc) {
            args.min_p = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--repeat-penalty") == 0 && i + 1 < argc) {
            args.repetition_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--frequency-penalty") == 0 && i + 1 < argc) {
            args.frequency_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--presence-penalty") == 0 && i + 1 < argc) {
            args.presence_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(arg, "--interactive") == 0) {
            args.interactive = true;
        } else if (std::strcmp(arg, "--device") == 0 && i + 1 < argc) {
            args.device = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--gpu-layers") == 0 && i + 1 < argc) {
            args.gpu_layers = std::atoi(argv[++i]);
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
        } else if (std::strcmp(arg, "--bench") == 0) {
            args.bench = true;
        } else if (std::strcmp(arg, "--bench-pp") == 0 && i + 1 < argc) {
            args.bench_pp = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--bench-reps") == 0 && i + 1 < argc) {
            args.bench_reps = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    return args;
}
