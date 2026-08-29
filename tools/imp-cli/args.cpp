#include "args.h"
#include "runtime/config.h"
#include "common/args_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

void print_usage(const char* prog) {
    // Two-step print: the only printf substitution is `prog` in the first
    // line. Everything else is static help text — emit it via fputs so any
    // `%` in option descriptions (e.g. "+4 % decode") is taken literally.
    // Previous single-fprintf invocation interpreted those as format specs
    // and corrupted the output ("+4 pct" was the workaround in PR #369).
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fputs(
            "Options:\n"
            "  --model <path>        Path to model file or HuggingFace repo ID (required)\n"
            "  --revision <rev>      HuggingFace model revision (branch, tag, or commit hash)\n"
            "  --config <path>       imp.conf path (default: ./imp.conf, ~/.config/imp/imp.conf)\n"
            "  --set <sec.key=val>   Override one imp.conf key (repeatable; an\n"
            "                        unknown key is an error, not a warning)\n"
            "  --prompt <text>       Input prompt for generation\n"
            "  --prompt-file <path>  Read the prompt from a file (whole content, verbatim).\n"
            "                        Long prompts exceed the OS argv limit (~128 KiB per\n"
            "                        argument, so ~32k tokens); mutually exclusive with --prompt\n"
            "  --max-tokens <n>      Maximum tokens to generate (default: 256)\n"
            "  --max-seq-len <n>     KV context ceiling in tokens (default: auto from VRAM)\n"
            "  --json                One JSON document on stdout, every human line on\n"
            "                        stderr. Works with --bench, --perplexity and --prompt;\n"
            "                        refused with --interactive\n"
            "  --mem-report          Print the full VRAM attribution table at init\n"
            "  --vram-budget <mb>    Hard per-process VRAM cap in MiB — size everything as if\n"
            "                        the GPU only had this much (multi-server on one GPU)\n"
            "  --min-kv-tokens <n>   Minimum KV capacity in tokens (default: auto)\n"
            "  --temperature <f>     Sampling temperature (default: 0.7)\n"
            "  --top-p <f>           Top-p (nucleus) sampling (default: 0.9)\n"
            "  --top-k <n>           Top-k sampling (default: 40)\n"
            "  --seed <n>            Random seed, -1 for random (default: -1)\n"
            "  --min-p <f>           Min-p sampling threshold (default: 0.0 = disabled)\n"
            "  --typical-p <f>       Locally typical sampling (default: 1.0 = disabled)\n"
            "  --repeat-penalty <f>  Repetition penalty (default: 1.0 = disabled)\n"
            "  --repeat-last-n <n>   Tokens the repetition penalty looks back over (default: 0 = all)\n"
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
            "  --kv-fp8              Opt in to FP8 E4M3 KV cache (halves KV memory).\n"
            "                        Default is FP16 KV — FP8 is not safe on every\n"
            "                        model (Mistral-Small-3.1 Q6_K, DeepSeek-R1,\n"
            "                        Qwen3.5-GDN Q8_0, Gemma-4 all hit a stride bug).\n"
            "  --kv-fp16             Force FP16 KV cache (opts out of the auto FP8\n"
            "                        upgrade for models with a kv-FP8 author hint).\n"
            "  --no-fp8-prefill      Disable auto FP8 weight cache for prefill.\n"
            "                        Use with --kv-fp16 --no-nvfp4 for full FP16\n"
            "                        path (fixes DeepSeek-R1 Q6_K garbage output).\n"
            "  --kv-int8             Use INT8 KV cache with dp4a attention (halves KV memory)\n"
            "  --kv-int4             Use INT4 KV cache (quarters KV memory)\n"
            "  --kv-nvfp4            Use NVFP4 KV cache (quarters KV memory; FP4 + E4M3 scales)\n"
            "  --kv-mxfp4            Use MXFP4-KV cache (quarters KV memory; FP4 + UE8M0 scales)\n"
            "  --ssm-fp16            Use FP16 for SSM h_state (saves ~50% SSM VRAM)\n"
            "  --no-cuda-graphs      Disable CUDA Graph capture for decode\n"
            "  --chat-template <t>   Chat template: auto, none, chatml, llama2, llama3, nemotron, gemma, "
            "deepseek_r1, phi\n"
            "  --prefill-chunk-size <n> Max tokens per prefill chunk (0 = single-chunk, default: per-arch)\n"
            "  --prefill-fp8         Use FP8 E4M3 weight cache for ~2x prefill throughput\n"
            "  --mtp-spec-decode <k> MTP drafting for the verify loop, chain length k (sidecar or embedded mtp.* head)\n"
            "  --decode-nvfp4        Force mode 1 (additive: FP8 prefill + NVFP4 decode caches).\n"
            "                        Auto-default for dense Q*_K (6-8 bit GGUF) on sm_120 since\n"
            "                        PR #367 — +4 % decode vs mode 2 at -9 % prefill on\n"
            "                        Qwen3-14B Q6_K.\n"
            "  --decode-nvfp4-only   Force mode 2 (replacement: NVFP4-only, no duplicate FP8 cache).\n"
            "                        Auto-default for native NVFP4 / MXFP4, MoE, GDN, sub-8-bit.\n"
            "                        Saves VRAM and gives faster prefill at -3 % decode — use\n"
            "                        for long-prompt workloads where prefill dominates wallclock.\n"
            "  --prefix-caching      Reuse KV cache blocks for shared token prefixes\n"
            "  --streaming-kv        Enable StreamingLLM smart KV cache (attention sinks + window)\n"
            "  --no-streaming-kv-auto  Disable auto-StreamingLLM when KV cache >90%% full\n"
            "  --stream-sinks <n>    Number of attention-sink tokens to always keep (default: 4)\n"
            "  --stream-window <n>   Sliding-window size (default: model's sliding_window)\n"
            "  --mxfp4-prefill       Use CUTLASS MXFP4 GEMM for prefill (sm_120, requires NVFP4)\n"
            "  --dual-path-quant     FP8 attention + NVFP4 FFN (higher quality attention, faster FFN)\n"
            "  --no-nvfp4            Disable NVFP4 decode cache (override auto-detection)\n"
            "  --stop <str>          Stop sequence (can specify multiple times, max 4)\n"
            "  --bench               Synthetic benchmark mode (like llama-bench)\n"
            "  --bench-pp <n>        Synthetic prompt token count (default: 512)\n"
            "  --bench-reps <n>      Repetitions to average (default: 3)\n"
            "  --perplexity <file>   Compute teacher-forced perplexity over a text file and exit\n"
            "  --calibrate <out>     With --perplexity: also write activation-calibration\n"
            "                        statistics to <out> (input for imp-quantize --calib)\n"
            "  --mmproj <path>       Path to vision encoder GGUF (mmproj) for Gemma-3/4;\n"
            "                        Qwen3-VL carries its tower in the checkpoint\n"
            "  --image <path>        Input image (needs a model with a vision tower).\n"
            "                        Repeat for several images (Qwen3-VL only)\n"
            "  --help                Show this help message\n",
            stderr);
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs args;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        // Shared imp-cli/imp-server flags first (#1209): a tool must not be able
        // to shadow one with a divergent local handler.
        if (parse_common_flag(args, argc, argv, i))
            continue;

        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            std::exit(0);
        } else if (std::strcmp(arg, "--perplexity") == 0 && i + 1 < argc) {
            args.perplexity_file = argv[++i];
        } else if (std::strcmp(arg, "--calibrate") == 0 && i + 1 < argc) {
            args.calibrate_out = argv[++i];
        } else if (std::strcmp(arg, "--prompt") == 0 && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (std::strcmp(arg, "--prompt-file") == 0 && i + 1 < argc) {
            args.prompt_file = argv[++i];
        } else if (std::strcmp(arg, "--max-seq-len") == 0 && i + 1 < argc) {
            args.max_seq_len = std::atoi(argv[++i]);
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
            args.repetition_penalty_set = true;
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
        } else if (std::strcmp(arg, "--kv-fp16") == 0) {
            // Force FP16 KV cache — opts out of the "auto" FP8 upgrade for
            // models that ship a kv_cache_quant_algo=FP8 hint.
            args.config_overrides.push_back("kv_cache.dtype=fp16");
        } else if (std::strcmp(arg, "--no-fp8-prefill") == 0) {
            args.config_overrides.push_back("attention.fp8_prefill=never");
        } else if (std::strcmp(arg, "--mtp-spec-decode") == 0 && i + 1 < argc) {
            args.mtp_spec_decode_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--prefill-fp8") == 0) {
            args.prefill_fp8 = true;
        } else if (std::strcmp(arg, "--prefix-caching") == 0) {
            args.prefix_caching = true;
        } else if (std::strcmp(arg, "--streaming-kv") == 0) {
            args.streaming_kv = true;
        } else if (std::strcmp(arg, "--no-streaming-kv-auto") == 0) {
            args.no_streaming_kv_auto = true;
        } else if (std::strcmp(arg, "--stream-sinks") == 0 && i + 1 < argc) {
            args.streaming_sinks = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--stream-window") == 0 && i + 1 < argc) {
            args.streaming_window = std::atoi(argv[++i]);
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
        } else if (std::strcmp(arg, "--image") == 0 && i + 1 < argc) {
            args.image_paths.push_back(argv[++i]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    // --prompt-file resolves here so the rest of the tool only ever sees
    // args.prompt. A missing/unreadable file is an error, not an empty prompt
    // (an empty prompt would silently fall through to "No prompt provided").
    if (!args.prompt_file.empty()) {
        if (!args.prompt.empty()) {
            fprintf(stderr, "--prompt and --prompt-file are mutually exclusive\n");
            std::exit(1);
        }
        FILE* f = std::fopen(args.prompt_file.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "--prompt-file: cannot open '%s'\n", args.prompt_file.c_str());
            std::exit(1);
        }
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz < 0) {
            fprintf(stderr, "--prompt-file: cannot size '%s'\n", args.prompt_file.c_str());
            std::fclose(f);
            std::exit(1);
        }
        args.prompt.resize(static_cast<size_t>(sz));
        const size_t got = sz > 0 ? std::fread(args.prompt.data(), 1, static_cast<size_t>(sz), f) : 0;
        std::fclose(f);
        if (got != static_cast<size_t>(sz)) {
            fprintf(stderr, "--prompt-file: short read on '%s' (%zu of %ld bytes)\n",
                    args.prompt_file.c_str(), got, sz);
            std::exit(1);
        }
        if (args.prompt.empty()) {
            fprintf(stderr, "--prompt-file: '%s' is empty\n", args.prompt_file.c_str());
            std::exit(1);
        }
    }

    return args;
}

// Bench / one-shot configuration pins, lifted out of main() (2026-08-29: the
// file sat exactly on the file-size gate's 800-line ceiling). Pure move - the
// policy and its rationale are unchanged, including that only an explicit
// `--set` wins over a pin: a stray imp.conf must not quietly redefine what
// tests/perf_baseline.json measures.
void apply_config_pins(imp::RuntimeConfig& runtime_cfg, const CliArgs& args) {
// Benchmark mode measures raw engine decode: MoE speculation would fold
// draft-acceptance luck + grouped-GEMM restart variance into the gated
// tg signal (dense spec stays as-is — measured neutral on the bench
// prompt). An explicit --set speculative.moe=… still wins.
if (args.bench) {
    bool user_set = false;
    for (const auto& ov : args.config_overrides)
        if (ov.rfind("speculative.moe", 0) == 0)
            user_set = true;
    if (!user_set)
        runtime_cfg.speculative.moe = false;
    // The suffix drafter is decidedly NOT bench-neutral (frequency-voted
    // adaptive drafts hit +170% tg128 on the bench prompt) — pin it to
    // the legacy scan so tests/perf_baseline.json keeps its raw-decode
    // semantics. An explicit --set speculative.suffix=… still wins.
    bool suffix_set = false;
    for (const auto& ov : args.config_overrides)
        if (ov.rfind("speculative.suffix=", 0) == 0)
            suffix_set = true;
    if (!suffix_set)
        runtime_cfg.speculative.suffix = false;
    // Recurrent snapshots (hybrid prefix caching) are dead weight in the
    // single-shot bench but their eager buffers shift the MoE expert
    // offload budget — pin them off so hybrid GGUF baselines are
    // unaffected. An explicit --set server.recurrent_snapshot_mb=… wins.
    bool snap_set = false;
    for (const auto& ov : args.config_overrides)
        if (ov.rfind("server.recurrent_snapshot_mb", 0) == 0)
            snap_set = true;
    if (!snap_set)
        runtime_cfg.server.recurrent_snapshot_mb = 0;
    // Hybrid (GDN/SSM) verify would fold draft-acceptance luck into the
    // gated tg signal exactly like the moe pin above — keep the
    // canonical baseline raw-decode. An explicit --set wins.
    bool hybrid_set = false;
    for (const auto& ov : args.config_overrides)
        if (ov.rfind("speculative.hybrid", 0) == 0)
            hybrid_set = true;
    if (!hybrid_set)
        runtime_cfg.speculative.hybrid = false;
    // MTP auto (speculative.mtp_k=-1) would draft during the gated bench on
    // any checkpoint that ships a head - the same "speculation folded into the
    // raw-decode signal" the pins above exist to prevent, and
    // tests/perf_baseline.json is what it would silently redefine (measured:
    // Qwen3.8-27B-NVFP4 --bench engaged the head before this pin). An explicit
    // --set speculative.mtp_k=... still wins.
    bool mtp_set = false;
    for (const auto& ov : args.config_overrides)
        if (ov.rfind("speculative.mtp_k", 0) == 0)
            mtp_set = true;
    if (!mtp_set && runtime_cfg.speculative.mtp_k < 0)
        runtime_cfg.speculative.mtp_k = 0;
    // Graph-captured verify (#847) changes verify-step timing (and pads
    // chunks) — keep the canonical baseline on the eager verify path.
    // An explicit --set speculative.capture=… still wins.
    bool capture_set = false;
    for (const auto& ov : args.config_overrides)
        if (ov.rfind("speculative.capture", 0) == 0)
            capture_set = true;
    if (!capture_set)
        runtime_cfg.speculative.capture = false;
    // SWA-aware KV sizing changes the KV layout and forces spec verify
    // eager on SWA models — keep the canonical baseline on full-length
    // KV. An explicit --set kv_cache.swa_sizing=… still wins.
    bool swa_set = false;
    for (const auto& ov : args.config_overrides)
        if (ov.rfind("kv_cache.swa_sizing", 0) == 0)
            swa_set = true;
    if (!swa_set)
        runtime_cfg.kv_cache.swa_sizing = "off";
}
// One-shot runs (--prompt / --bench) never re-see a prefix: the process
// exits after a single generation, so prefix caching only costs hashing
// and blocks the swa_sizing=auto KV savings on SWA models. Interactive
// mode keeps it (turn N+1 reuses turn N's prefix). --prefix-caching or
// an explicit --set server.prefix_cache=… still wins.
if (!args.interactive && !args.prefix_caching) {
    bool pc_set = false;
    for (const auto& ov : args.config_overrides)
        if (ov.rfind("server.prefix_cache", 0) == 0)
            pc_set = true;
    if (!pc_set)
        runtime_cfg.server.prefix_cache = false;
}
}
