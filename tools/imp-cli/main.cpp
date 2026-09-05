#include "api/imp_internal.h"
#include "common/exit_codes.h"
#include "common/mtp_auto.h"
#include "common/json_out.h"
#include "json_report.h"
#include "modes.h"
#include "args.h"
#include "model/chat_template.h"
#include "model/image_placeholders.h"
#include "model/hf_hub.h"
#include "model/tokenizer.h"
#include <sys/stat.h>
#include "runtime/presets.h"
#include "runtime/config.h"
#include "core/process_diag.h"
#include "runtime/process_diag_install.h"
#include "runtime/engine.h"
#include "memory/vram_query.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    CliArgs args = parse_args(argc, argv);

    // #1583: with --json, stdout belongs to the single JSON document. Reserve
    // it here, before the first print, and route every human line to stderr.
    // Interactive mode has no single document to emit, so it declines.
    if (args.json_out) {
        if (args.interactive) {
            fprintf(stderr,
                    "Error: --json has no meaning with --interactive (the output is a "
                    "token stream, not one document)\n");
            return imp::tools::exit_code_for(IMP_ERROR_INVALID_ARG);
        }
        imp_tools::json_stdout_reserve();
    }

    // Load imp.conf (if present) + apply --set overrides, then stash for
    // Engine::init to pick up (Phase 5 Track D follow-up: replaces the
    // RuntimeConfig::install() process-wide singleton).
    std::vector<std::string> rejected_overrides;
    imp::RuntimeConfig runtime_cfg =
        imp::RuntimeConfig::load(args.config_path, args.config_overrides, &rejected_overrides);
    if (!rejected_overrides.empty()) {
        // Silently ignoring these is how a benchmark ends up measuring a
        // configuration nobody asked for: `--set gemm.deterministic=true` sat
        // in the AWQ reproduction harness doing nothing at all.
        for (const auto& bad : rejected_overrides)
            fprintf(stderr, "Error: --set %s\n", bad.c_str());
        fprintf(stderr, "See imp.conf.example for the key names.\n");
        return 1;
    }
    apply_config_pins(runtime_cfg, args);
    // `[calibration] out_path` is the fallback when --calibrate carries no path.
    // The key was parsed and documented and read by nothing, so an operator who
    // set it in imp.conf got no file and no warning (debt ledger item 7).
    args.calibrate_out = resolve_calibration_out(args.calibrate_out, runtime_cfg.calibration.out_path);

    // --calibrate is only meaningful alongside a corpus pass: the statistics
    // are what a forward pass saw, and --perplexity is the pass that walks a
    // whole corpus without sampling.
    if (!args.calibrate_out.empty()) {
        if (args.perplexity_file.empty()) {
            fprintf(stderr, "Error: --calibrate requires --perplexity <corpus>\n");
            return 1;
        }
        runtime_cfg.calibration.enabled = true;
        // A calibration file that varies run to run makes the checkpoint built
        // from it unreproducible, and the variance is NOT small: two runs of
        // this exact command differed on 94% of the recorded floats (up to
        // 0.5% each), which moved the quantized model's perplexity by ~1.6%.
        // runtime.deterministic_gemm makes it bit-identical, so calibration
        // forces it rather than offering it.
        if (!runtime_cfg.runtime.deterministic_gemm) {
            runtime_cfg.runtime.deterministic_gemm = true;
            fprintf(stderr,
                    "--calibrate: forcing runtime.deterministic_gemm=true "
                    "(a non-reproducible calibration file is not worth having)\n");
        }
    }
    // Cache the few diagnostic / runtime-mode flags that are read from free
    // functions (kernel diagnostics, CUDA-graph capture mode, PDL gate)
    // before set_pending_runtime_config() consumes the value.
    imp::process_diag_install(runtime_cfg);
    imp::set_pending_runtime_config(runtime_cfg);

    if (args.model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    printf("IMP Inference Engine %s\n", imp_version());

    // Resolve model path: supports local files, directories, and HuggingFace repo IDs.
    // Auto-detect format: directories with .safetensors → SafeTensors, else GGUF.
    std::string resolved_model = args.model_path;
    ImpModelFormat format = IMP_FORMAT_GGUF;

    // Check if path is a directory with SafeTensors files
    {
        struct stat st {};
        if (stat(args.model_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            std::string index = args.model_path + "/model.safetensors.index.json";
            std::string single = args.model_path + "/model.safetensors";
            struct stat idx_st {};
            if (stat(index.c_str(), &idx_st) == 0 || stat(single.c_str(), &idx_st) == 0) {
                format = IMP_FORMAT_SAFETENSORS;
            }
        }
    }

    if (format == IMP_FORMAT_GGUF) {
        resolved_model = imp::resolve_model_gguf(args.model_path, args.revision);
        if (resolved_model.empty()) {
            // The most common failure of all, and the one a caller most wants
            // to distinguish: a path that is not there. It never reached
            // imp_model_load_ex, so it needs its own code rather than the
            // generic 1 (#1585).
            fprintf(stderr, "Failed to resolve model: %s\n", args.model_path.c_str());
            return imp::tools::exit_code_for(IMP_ERROR_FILE_NOT_FOUND);
        }
        if (resolved_model != args.model_path) {
            printf("Resolved model: %s -> %s\n", args.model_path.c_str(), resolved_model.c_str());
        }
    }

    printf("Loading model: %s (%s)\n", resolved_model.c_str(),
           format == IMP_FORMAT_SAFETENSORS ? "SafeTensors" : "GGUF");

    auto t_init_start = std::chrono::high_resolution_clock::now();

    ImpModel model = nullptr;
    // Only load the MTP head sidecar (~1.57 GiB BF16 on Qwen3.6) when the user
    // actually requested MTP spec-decode. Otherwise it is dead VRAM.
    // Both spellings count: the startup hint recommends --set
    // speculative.mtp_k=2, and until this line the CLI silently ignored it.
    // speculative.mtp_k is a tri-state: -1 auto, 0 off, >0 fixed. The CLI is
    // always single-stream (config.max_batch_size = 1 below), so auto engages
    // here whenever the checkpoint ships a head; --mtp-spec-decode still wins.
    int mtp_k = args.mtp_spec_decode_k > 0
                    ? args.mtp_spec_decode_k
                    : imp::tools::mtp_auto_request_k(runtime_cfg, /*configured_batch=*/1);
    ImpError err = imp_model_load_ex(resolved_model.c_str(), format,
                                     /*load_mtp_head=*/mtp_k > 0, &model);
    if (err != IMP_SUCCESS) {
        fprintf(stderr, "Error loading model: %s\n", imp_error_string(err));
        return imp::tools::exit_code_for(err);
    }
    // What the load produced decides the pair: a checkpoint without a head must
    // not end up with ngram off and nothing drafting (mtp_auto_after_load).
    mtp_k = imp::tools::mtp_auto_after_load(
        runtime_cfg, mtp_k, model->model->mtp_.has_value() && model->model->mtp_->loaded,
        args.mtp_spec_decode_k);

    ImpConfig config = imp_config_default();

    // Sampling defaults: CLI flag > generation_config.json (SafeTensors only) >
    // arch-family preset. Author-shipped values from generation_config.json are
    // signalled by sentinel >= 0; sentinel <0 falls through to the family preset.
    imp::SamplingDefaults sampling = imp::get_sampling_defaults(model->model->config().arch);
    const auto& gen = model->model->generation_config();
    if (gen.temperature >= 0.0f)
        sampling.temperature = gen.temperature;
    if (gen.top_p >= 0.0f)
        sampling.top_p = gen.top_p;
    if (gen.top_k >= 0)
        sampling.top_k = gen.top_k;

    // CLI flags override auto-detection (only when explicitly set)
    config.device_id = args.device;
    // CLI is single-request — always cap batch size to 1
    config.max_batch_size = 1;
    // max_seq_len: 0 = auto-detect in engine (from model metadata + VRAM)
    if (args.max_seq_len > 0)
        config.max_seq_len = args.max_seq_len;
    if (args.vram_budget_mb > 0)
        config.vram_budget_mb = args.vram_budget_mb;
    if (args.min_kv_tokens > 0)
        config.min_kv_tokens = args.min_kv_tokens;
    config.gpu_layers = args.gpu_layers;
    if (args.kv_fp8)
        config.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;
    if (args.kv_int8)
        config.kv_cache_dtype = IMP_DTYPE_INT8;
    if (args.kv_int4)
        config.kv_cache_dtype = IMP_DTYPE_INT4;
    if (args.kv_nvfp4)
        config.kv_cache_dtype = IMP_DTYPE_NVFP4;
    if (args.kv_mxfp4)
        config.kv_cache_dtype = IMP_DTYPE_MXFP4_KV;
    if (args.ssm_fp16)
        config.ssm_state_dtype = IMP_DTYPE_FP16;
    if (args.no_cuda_graphs)
        config.enable_cuda_graphs = 0;
    if (args.prefill_chunk_size >= 0)
        config.prefill_chunk_size = args.prefill_chunk_size;
    if (args.prefill_fp8)
        config.use_fp8_prefill = 1;
    if (args.mxfp4_prefill)
        config.use_mxfp4_prefill = 1;
    if (args.dual_path_quant)
        config.dual_path_quant = 1;
    // --prefix-caching, else imp.conf ([server] prefix_cache, default on). The
    // engine used to OR the imp.conf value in for every embedder; it no longer
    // does, so the CLI states its own choice (#1299).
    config.use_prefix_caching = (args.prefix_caching || runtime_cfg.server.prefix_cache) ? 1 : 0;
    if (args.streaming_kv) {
        config.streaming_kv_enabled = 1;
        config.streaming_kv_n_sinks = args.streaming_sinks;
        config.streaming_kv_window = args.streaming_window;
    }
    if (args.no_streaming_kv_auto)
        config.streaming_kv_auto = 0;
    config.use_nvfp4_decode = args.decode_nvfp4;
    if (!args.mmproj_path.empty())
        config.mmproj_path = args.mmproj_path.c_str();

    // In bench mode, ensure KV cache matches what the benchmark needs.
    // Raises max_seq_len for long-context benchmarks, caps it for short ones.
    if (args.bench) {
        // Headroom: at least 256 tokens, at least 12.5% of the bench shape. The
        // StreamingLLM valve (engine_scheduler.cpp) fires on an F16 KV cache
        // when under 10% of the pool is free; with a flat +256 an F16 model at
        // pp >= ~2.3k benched into the valve, the first decode step produced
        // no token and the bench printed 0 tok/s (Llama-3.2-3B-Q8_0 at pp
        // 8192: "0/536 blocks free", 2026-09-03). pp512 + tg128 stays at 896,
        // so the perf-gate pool is unchanged.
        const int bench_shape = args.bench_pp + args.max_tokens;
        int bench_need = bench_shape + std::max(256, bench_shape / 8);
        if (config.max_seq_len != bench_need) {
            config.max_seq_len = bench_need;
        }
        // Single-request benchmark — no batching needed
        config.max_batch_size = 1;
    }

    // Perplexity mode: read + tokenize the corpus up front so we can size KV and
    // force single-chunk prefill (all-position hidden must survive for the PPL pass).
    std::vector<int32_t> ppl_tokens;
    if (!args.perplexity_file.empty()) {
        FILE* f = std::fopen(args.perplexity_file.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "Error: cannot open --perplexity file %s\n", args.perplexity_file.c_str());
            imp_model_free(model);
            return 1;
        }
        std::string text;
        char buf[4096];
        size_t r;
        while ((r = std::fread(buf, 1, sizeof(buf), f)) > 0)
            text.append(buf, r);
        std::fclose(f);
        int vocab_size = imp_model_vocab_size(model);
        (void)vocab_size;
        int max_tok = static_cast<int>(text.size()) + 16;
        ppl_tokens.resize(max_tok);
        int n_tok = 0;
        ImpError te = imp_tokenize(model, text.c_str(), ppl_tokens.data(), &n_tok, max_tok);
        if (te != IMP_SUCCESS || n_tok < 2) {
            fprintf(stderr, "Error tokenizing perplexity corpus (%s, n=%d)\n", imp_error_string(te), n_tok);
            imp_model_free(model);
            return 1;
        }
        ppl_tokens.resize(n_tok);
        // BOS-dependent families (Gemma especially) need the BOS prepended or
        // the teacher-forced NLL measures an out-of-distribution sequence
        // (gemma-3-12b read PPL ~100 instead of ~10 without it).
        int32_t bos = imp_model_bos_token(model);
        if (bos >= 0 && ppl_tokens[0] != bos) {
            ppl_tokens.insert(ppl_tokens.begin(), bos);
            n_tok++;
        }
        // diagnostics.dump_tokens: print the FULL corpus token stream (one id
        // per line on stderr) — cross-engine tokenizer forensics (#657) diffs
        // this against `llama-tokenize --ids` output.
        if (runtime_cfg.diagnostics.dump_tokens) {
            fprintf(stderr, "[DUMP_PPL_TOKENS] n=%d\n", n_tok);
            for (int ti = 0; ti < n_tok; ti++)
                fprintf(stderr, "TOK %d %d\n", ti, ppl_tokens[ti]);
        }
        // Chunked prefill is the DEFAULT here since the engine-side capture
        // became chunk-aware (#553): per-chunk NLL accumulation makes the
        // teacher-forced score independent of chunking, and the chunked
        // rectangular cuBLAS path is the attention-correctness reference for
        // long corpora (the single-chunk route falls into the FMHA/WMMA
        // family beyond the S-matrix cap — the #566 WMMA hd=256 remnant).
        // An explicit --prefill-chunk-size (incl. 0) still wins.
        config.max_batch_size = 1;
        config.max_seq_len = n_tok + 16;
        fprintf(stderr, "Perplexity: %d tokens from %s\n", n_tok, args.perplexity_file.c_str());
    }

    ImpContext ctx = nullptr;
    err = imp_context_create(model, &config, &ctx);
    if (err == IMP_SUCCESS && mtp_k > 0) {
        ImpError mtp_err = imp_enable_mtp_spec_decode(ctx, mtp_k);
        if (mtp_err != IMP_SUCCESS) {
            fprintf(stderr, "Warning: MTP spec-decode k=%d failed (%s); continuing without spec-decode\n",
                    mtp_k, imp_error_string(mtp_err));
        }
    }
    if (err != IMP_SUCCESS) {
        fprintf(stderr, "Error creating context: %s\n", imp_error_string(err));
        imp_model_free(model);
        return imp::tools::exit_code_for(err);
    }

    auto t_init_end = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();
    fprintf(stderr, "Init: %.2f ms (model load + engine setup)\n", init_ms);

    ImpGenerateParams params = imp_generate_params_default();
    // Use model-family sampling defaults unless explicitly overridden via CLI flags
    params.temperature = args.temperature_set ? args.temperature : sampling.temperature;
    params.top_p = args.top_p_set ? args.top_p : sampling.top_p;
    params.top_k = args.top_k_set ? args.top_k : sampling.top_k;
    params.max_tokens = args.max_tokens;
    params.seed = args.seed;
    params.min_p = args.min_p;
    params.typical_p = args.typical_p;
    params.repetition_penalty = args.repetition_penalty_set
                                    ? args.repetition_penalty
                                    : (gen.repetition_penalty >= 0.0f ? gen.repetition_penalty
                                                                      : args.repetition_penalty);
    params.frequency_penalty = args.frequency_penalty;
    params.presence_penalty = args.presence_penalty;
    params.repeat_last_n = args.repeat_last_n;
    params.dry_multiplier = args.dry_multiplier;
    params.dry_base = args.dry_base;
    params.dry_allowed_length = args.dry_allowed_length;
    params.dry_penalty_last_n = args.dry_penalty_last_n;
    params.mirostat = args.mirostat;
    params.mirostat_tau = args.mirostat_tau;
    params.mirostat_eta = args.mirostat_eta;

    // Determine chat template override from --chat-template flag
    if (args.chat_template == "none") {
        params.apply_chat_template = 0;
    }

    if (!args.perplexity_file.empty()) {
        const int rc = imp_cli::run_perplexity(ctx, args, ppl_tokens, resolved_model);
        imp_context_free(ctx);
        imp_model_free(model);
        return rc;
    }

    const int rc = args.bench         ? imp_cli::run_bench(ctx, model, args, resolved_model)
                   : args.interactive ? imp_cli::run_interactive(ctx, model, args, params)
                                      : imp_cli::run_oneshot(ctx, model, args, params, resolved_model);

    imp_context_free(ctx);
    imp_model_free(model);
    return rc;
}
