#include "imp/imp.h"
#include "args.h"
#include "model/chat_template.h"
#include "model/tokenizer.h"
#include "runtime/engine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Access internal engine from opaque context handle
// (imp-cli links against imp with PRIVATE src/ include access)
struct ImpModel_T {
    std::shared_ptr<imp::Model> model;
};

struct ImpContext_T {
    ImpModel model_handle = nullptr;
    std::unique_ptr<imp::Engine> engine;
    std::shared_ptr<imp::Request> active_request;
};

int main(int argc, char** argv) {
    CliArgs args = parse_args(argc, argv);

    if (args.model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    printf("IMP Inference Engine %s\n", imp_version());
    printf("Loading model: %s\n", args.model_path.c_str());

    auto t_init_start = std::chrono::high_resolution_clock::now();

    ImpModel model = nullptr;
    ImpError err = imp_model_load(args.model_path.c_str(), IMP_FORMAT_GGUF, &model);
    if (err != IMP_SUCCESS) {
        fprintf(stderr, "Error loading model: %s\n", imp_error_string(err));
        return 1;
    }

    ImpConfig config = imp_config_default();
    config.device_id = args.device;
    config.max_batch_size = 1;
    config.max_seq_len = 4096;
    config.gpu_layers = args.gpu_layers;
    if (args.kv_fp8) config.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;
    if (args.kv_int8) config.kv_cache_dtype = IMP_DTYPE_INT8;
    if (args.ssm_fp16) config.ssm_state_dtype = IMP_DTYPE_FP16;
    // CUDA graphs enabled by default in imp_config_default(); --no-cuda-graphs can disable
    if (args.no_cuda_graphs) config.enable_cuda_graphs = 0;
    config.prefill_chunk_size = args.prefill_chunk_size;

    ImpContext ctx = nullptr;
    err = imp_context_create(model, &config, &ctx);
    if (err != IMP_SUCCESS) {
        fprintf(stderr, "Error creating context: %s\n", imp_error_string(err));
        imp_model_free(model);
        return 1;
    }

    auto t_init_end = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();
    fprintf(stderr, "Init: %.2f ms (model load + engine setup)\n", init_ms);

    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = args.temperature;
    params.top_p = args.top_p;
    params.top_k = args.top_k;
    params.max_tokens = args.max_tokens;
    params.seed = args.seed;
    params.min_p = args.min_p;
    params.typical_p = args.typical_p;
    params.repetition_penalty = args.repetition_penalty;
    params.frequency_penalty = args.frequency_penalty;
    params.presence_penalty = args.presence_penalty;
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

    if (args.bench) {
        // Synthetic benchmark mode (matches llama-bench methodology)
        int vocab_size = imp_model_vocab_size(model);
        std::vector<int32_t> tokens(args.bench_pp);
        for (int i = 0; i < args.bench_pp; i++)
            tokens[i] = i % vocab_size;

        int tg_tokens = args.max_tokens;

        // Greedy decode params for deterministic benchmarking
        ImpGenerateParams bench_params = imp_generate_params_default();
        bench_params.temperature = 0.0f;
        bench_params.ignore_eos = 1;  // Don't stop on EOS during benchmark
        // +1 because imp_prefill already produces the first output token;
        // without this the request hits max_tokens one decode step early.
        bench_params.max_tokens = tg_tokens + 1;

        fprintf(stderr, "Benchmark: pp=%d, tg=%d, reps=%d\n", args.bench_pp, tg_tokens, args.bench_reps);

        // Warmup: 1 full prefill+decode cycle (discarded)
        fprintf(stderr, "Warmup...\n");
        imp_context_reset(ctx);
        imp_prefill(ctx, tokens.data(), args.bench_pp);
        for (int s = 0; s < tg_tokens; s++) {
            int32_t tok = 0;
            imp_decode_step(ctx, &bench_params, &tok);
        }

        // PP benchmark
        double pp_total_ms = 0;
        for (int rep = 0; rep < args.bench_reps; rep++) {
            imp_context_reset(ctx);
            auto t0 = std::chrono::high_resolution_clock::now();
            err = imp_prefill(ctx, tokens.data(), args.bench_pp);
            auto t1 = std::chrono::high_resolution_clock::now();
            if (err != IMP_SUCCESS) {
                fprintf(stderr, "Prefill error on rep %d: %s\n", rep, imp_error_string(err));
                break;
            }
            pp_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        // TG benchmark
        double tg_total_ms = 0;
        for (int rep = 0; rep < args.bench_reps; rep++) {
            imp_context_reset(ctx);
            err = imp_prefill(ctx, tokens.data(), args.bench_pp);
            if (err != IMP_SUCCESS) {
                fprintf(stderr, "Prefill error on tg rep %d: %s\n", rep, imp_error_string(err));
                break;
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < tg_tokens; s++) {
                int32_t tok = 0;
                err = imp_decode_step(ctx, &bench_params, &tok);
                if (err != IMP_SUCCESS) break;
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            if (err != IMP_SUCCESS) {
                fprintf(stderr, "Decode error on rep %d: %s\n", rep, imp_error_string(err));
                break;
            }
            tg_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double pp_avg_ms = pp_total_ms / args.bench_reps;
        double tg_avg_ms = tg_total_ms / args.bench_reps;
        double pp_toks = (pp_avg_ms > 0) ? (args.bench_pp / (pp_avg_ms / 1000.0)) : 0;
        double tg_toks = (tg_avg_ms > 0) ? (tg_tokens / (tg_avg_ms / 1000.0)) : 0;

        fprintf(stderr, "pp %5d tokens  avg %8.2f ms  (%7.2f tok/s)  [%d reps]\n",
                args.bench_pp, pp_avg_ms, pp_toks, args.bench_reps);
        fprintf(stderr, "tg %5d tokens  avg %8.2f ms  (%7.2f tok/s)  [%d reps]\n",
                tg_tokens, tg_avg_ms, tg_toks, args.bench_reps);
    } else if (args.interactive) {
        // Multi-turn interactive mode using token-level API with chat template
        imp::Tokenizer* tok = model->model->tokenizer();
        const imp::ChatTemplate& engine_tpl = ctx->engine->chat_template();

        // Resolve effective chat template: CLI override or engine-detected
        imp::ChatTemplate chat_tpl;
        bool have_template = false;

        if (args.chat_template == "none") {
            // No template
        } else if (args.chat_template != "auto") {
            // Explicit override from CLI
            auto family = imp::ChatTemplate::parse_family(args.chat_template);
            if (family != imp::ChatTemplateFamily::RAW) {
                have_template = chat_tpl.init(family, *tok);
            }
        } else {
            // Use engine-detected template
            if (!engine_tpl.is_raw()) {
                chat_tpl = engine_tpl;
                have_template = true;
            }
        }

        if (have_template) {
            printf("Chat template: %s\n", imp::chat_template_family_name(chat_tpl.family()));
        } else {
            printf("No chat template (raw mode)\n");
        }

        printf("Interactive mode. Type 'quit' to exit.\n");

        std::vector<imp::ChatMessage> history;
        char line[4096];

        while (true) {
            printf("\n> ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;

            // Trim trailing newline
            size_t len = std::strlen(line);
            if (len > 0 && line[len - 1] == '\n') line[len - 1] = '\0';

            std::string input(line);
            if (input.empty() || input == "quit" || input == "exit") break;

            if (have_template) {
                // Multi-turn: append user message and apply full template
                history.push_back({"user", input});
                std::vector<int32_t> tokens = chat_tpl.apply(*tok, history);

                // Reset context for fresh KV cache
                imp_context_reset(ctx);

                // Prefill with templated tokens
                err = imp_prefill(ctx, tokens.data(), static_cast<int>(tokens.size()));
                if (err != IMP_SUCCESS) {
                    fprintf(stderr, "Prefill error: %s\n", imp_error_string(err));
                    history.pop_back();
                    continue;
                }

                // Decode token by token
                std::vector<int32_t> output_ids;
                std::string response;
                for (int step = 0; step < params.max_tokens; step++) {
                    int32_t token = 0;
                    err = imp_decode_step(ctx, &params, &token);
                    if (err != IMP_SUCCESS) break;

                    // Check stop tokens
                    if (token == tok->eos_id()) break;
                    bool is_stop = false;
                    for (int32_t stop_id : chat_tpl.stop_token_ids()) {
                        if (token == stop_id) { is_stop = true; break; }
                    }
                    if (is_stop) break;

                    output_ids.push_back(token);
                    std::string piece = tok->decode_token(token);
                    printf("%s", piece.c_str());
                    fflush(stdout);
                }
                printf("\n");

                response = tok->decode(output_ids);
                history.push_back({"assistant", response});
            } else {
                // Raw mode: no history, just generate
                imp_context_reset(ctx);
                char output[8192];
                size_t output_len = 0;
                err = imp_generate(ctx, input.c_str(), &params, output, sizeof(output), &output_len);
                if (err != IMP_SUCCESS) {
                    fprintf(stderr, "Generation error: %s\n", imp_error_string(err));
                    continue;
                }
                printf("%.*s\n", (int)output_len, output);
            }
        }
    } else {
        // Single-shot mode with timing
        if (args.prompt.empty()) {
            fprintf(stderr, "No prompt provided. Use --prompt or --interactive\n");
        } else {
            imp::Tokenizer* tok = model->model->tokenizer();
            const imp::ChatTemplate& engine_tpl = ctx->engine->chat_template();

            // Resolve chat template
            imp::ChatTemplate chat_tpl;
            bool have_template = false;
            if (args.chat_template == "none" || !params.apply_chat_template) {
                // No template
            } else if (args.chat_template != "auto") {
                auto family = imp::ChatTemplate::parse_family(args.chat_template);
                if (family != imp::ChatTemplateFamily::RAW) {
                    have_template = chat_tpl.init(family, *tok);
                }
            } else if (!engine_tpl.is_raw()) {
                chat_tpl = engine_tpl;
                have_template = true;
            }

            // Tokenize prompt
            std::vector<int32_t> tokens;
            if (have_template) {
                std::vector<imp::ChatMessage> msgs = {{"user", args.prompt}};
                tokens = chat_tpl.apply(*tok, msgs);
            } else {
                tokens = tok->encode(args.prompt);
                // Prepend BOS when the tokenizer requires it (e.g. Gemma)
                if (tok->add_bos()) {
                    tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));
                }
            }
            int n_prompt_tokens = static_cast<int>(tokens.size());

            // Prefill with timing
            auto t_prefill_start = std::chrono::high_resolution_clock::now();
            err = imp_prefill(ctx, tokens.data(), n_prompt_tokens);
            auto t_prefill_end = std::chrono::high_resolution_clock::now();
            if (err != IMP_SUCCESS) {
                fprintf(stderr, "Prefill error: %s\n", imp_error_string(err));
                imp_context_free(ctx);
                imp_model_free(model);
                return 1;
            }

            // Decode with timing
            auto t_decode_start = std::chrono::high_resolution_clock::now();
            std::vector<int32_t> output_ids;
            for (int step = 0; step < params.max_tokens; step++) {
                int32_t token = 0;
                err = imp_decode_step(ctx, &params, &token);
                if (err != IMP_SUCCESS) break;

                if (token == tok->eos_id()) break;
                if (have_template) {
                    bool is_stop = false;
                    for (int32_t stop_id : chat_tpl.stop_token_ids()) {
                        if (token == stop_id) { is_stop = true; break; }
                    }
                    if (is_stop) break;
                }

                output_ids.push_back(token);
                std::string piece = tok->decode_token(token);
                if (step < 10) fprintf(stderr, "[tok=%d '%s'] ", token, piece.c_str());
                printf("%s", piece.c_str());
                fflush(stdout);
            }
            auto t_decode_end = std::chrono::high_resolution_clock::now();
            printf("\n");

            int n_output_tokens = static_cast<int>(output_ids.size());
            double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
            double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
            double total_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_prefill_start).count();

            double pp_toks = (prefill_ms > 0) ? (n_prompt_tokens / (prefill_ms / 1000.0)) : 0;
            double tg_toks = (decode_ms > 0 && n_output_tokens > 1)
                ? ((n_output_tokens - 1) / (decode_ms / 1000.0)) : 0;

            fprintf(stderr, "\n");
            fprintf(stderr, "pp %5d tokens in %8.2f ms  (%7.2f tok/s)\n", n_prompt_tokens, prefill_ms, pp_toks);
            fprintf(stderr, "tg %5d tokens in %8.2f ms  (%7.2f tok/s)\n", n_output_tokens, decode_ms, tg_toks);
            fprintf(stderr, "total   %8.2f ms\n", total_ms);

            // Benchmark using Engine::generate() (conditional graph loop) for comparison.
            // This eliminates per-step host overhead — shows true GPU-limited throughput.
            if (std::getenv("IMP_BENCH_GENERATE")) {
                // Reset context for fresh generation
                imp_context_reset(ctx);

                // Use Engine::generate() directly for accurate timing
                imp::Engine* engine = ctx->engine.get();
                auto t_gen_start = std::chrono::high_resolution_clock::now();
                std::string gen_result = engine->generate(
                    args.prompt, params.max_tokens,
                    params.temperature, params.top_p, params.top_k, params.seed,
                    have_template);
                auto t_gen_end = std::chrono::high_resolution_clock::now();

                // Count output tokens by encoding the result
                auto gen_toks = tok->encode(gen_result);
                int gen_n = static_cast<int>(gen_toks.size());
                double gen_total_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
                // Estimate decode time: total - prefill (reuse prefill timing from above)
                double gen_decode_ms = gen_total_ms - prefill_ms;
                double gen_toks_s = (gen_decode_ms > 0 && gen_n > 0)
                    ? (gen_n / (gen_decode_ms / 1000.0)) : 0;
                fprintf(stderr, "graph-loop: %d tg tokens in %.2f ms (%.2f tok/s, %.2f ms total)\n",
                        gen_n, gen_decode_ms, gen_toks_s, gen_total_ms);
            }
        }
    }

    imp_context_free(ctx);
    imp_model_free(model);
    return 0;
}
