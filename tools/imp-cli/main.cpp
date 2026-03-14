#include "api/imp_internal.h"
#include "args.h"
#include "model/chat_template.h"
#include "model/tokenizer.h"
#include "runtime/presets.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    CliArgs args = parse_args(argc, argv);

    // Load presets (TOML file or built-in fallback)
    imp::load_presets(args.presets_file);

    // Handle --preset list
    if (args.preset == "list") {
        imp::print_presets();
        return 0;
    }

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

    // Resolve preset: explicit --preset flag > auto-detect from filename
    const imp::PresetConfig* preset = nullptr;
    if (args.preset == "none") {
        // Explicitly disabled
    } else if (!args.preset.empty()) {
        preset = imp::find_preset(args.preset);
        if (!preset) {
            fprintf(stderr, "Unknown preset: %s (use --preset list to see available presets)\n",
                    args.preset.c_str());
            return 1;
        }
    } else {
        preset = imp::detect_preset(args.model_path);
    }

    if (preset) {
        imp::apply_preset(preset, config);
        fprintf(stderr, "Preset: %s\n", preset->description.c_str());
    }

    // CLI flags override preset values (only when explicitly set)
    config.device_id = args.device;
    // CLI is single-request — always cap batch size to 1 (preset values are for server).
    config.max_batch_size = 1;
    if (!preset) {
        config.max_seq_len = 4096;
    }
    config.gpu_layers = args.gpu_layers;
    if (args.kv_fp8) config.kv_cache_dtype = IMP_DTYPE_FP8_E4M3;
    if (args.kv_int8) config.kv_cache_dtype = IMP_DTYPE_INT8;
    if (args.kv_int4) config.kv_cache_dtype = IMP_DTYPE_INT4;
    if (args.ssm_fp16) config.ssm_state_dtype = IMP_DTYPE_FP16;
    if (args.no_cuda_graphs) config.enable_cuda_graphs = 0;
    if (args.prefill_chunk_size > 0) config.prefill_chunk_size = args.prefill_chunk_size;
    if (args.prefill_fp8) config.use_fp8_prefill = 1;
    if (args.mxfp4_prefill) config.use_mxfp4_prefill = 1;
    if (args.prefix_caching) config.use_prefix_caching = 1;
    if (args.decode_nvfp4 != -1 || !preset)
        config.use_nvfp4_decode = args.decode_nvfp4;
    if (!args.mmproj_path.empty())
        config.mmproj_path = args.mmproj_path.c_str();
    if (args.self_speculative) {
        config.enable_self_speculative = 1;
        config.self_spec_k = args.self_spec_k;
        config.self_spec_exit_layer = args.self_spec_exit_layer;
        config.self_spec_skip_n = args.self_spec_skip_n;
    }

    // In bench mode, cap KV cache to what the benchmark actually needs.
    // Prevents OOM when presets specify large max_seq_len (e.g. 131072).
    if (args.bench) {
        int bench_need = args.bench_pp + args.max_tokens + 256;  // +256 headroom
        if (config.max_seq_len > bench_need) {
            config.max_seq_len = bench_need;
        }
        // Single-request benchmark — no batching needed
        config.max_batch_size = 1;
    }

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
    // Use preset sampling values unless explicitly overridden via CLI flags
    params.temperature = (preset && !args.temperature_set) ? config.temperature : args.temperature;
    params.top_p = (preset && !args.top_p_set) ? config.top_p : args.top_p;
    params.top_k = (preset && !args.top_k_set) ? config.top_k : args.top_k;
    params.max_tokens = args.max_tokens;
    params.seed = args.seed;
    params.min_p = args.min_p;
    params.typical_p = args.typical_p;
    params.repetition_penalty = args.repetition_penalty;
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
        // Interactive/agentic defaults to 16384 max tokens (needs headroom for
        // long reasoning chains, code generation, and multi-step tool use)
        if (!args.max_tokens_set) {
            params.max_tokens = 16384;
        }
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
        if (ctx->engine->has_vision()) {
            printf("Vision enabled. Use '/image <path>' to load an image.\n");
        }

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

            // Handle /image command
            if (input.rfind("/image ", 0) == 0) {
                std::string img_path = input.substr(7);
                err = imp_set_image(ctx, img_path.c_str());
                if (err != IMP_SUCCESS) {
                    fprintf(stderr, "Error loading image: %s\n", imp_error_string(err));
                } else {
                    printf("Image loaded: %s\n", img_path.c_str());
                }
                continue;
            }

            if (have_template) {
                // Multi-turn: append user message and apply full template
                history.push_back({"user", input});
                std::vector<int32_t> tokens;
                if (ctx->engine->has_vision_input()) {
                    tokens = chat_tpl.apply_with_image(*tok, history, 256);
                } else {
                    tokens = chat_tpl.apply(*tok, history);
                }

                // Reset context for fresh KV cache
                imp_context_reset(ctx);

                // Prefill with templated tokens
                err = imp_prefill(ctx, tokens.data(), static_cast<int>(tokens.size()));
                if (err != IMP_SUCCESS) {
                    fprintf(stderr, "Prefill error: %s\n", imp_error_string(err));
                    history.pop_back();
                    continue;
                }

                // Capture the first token produced during prefill
                // (engine->step() generates it as part of the prefill pass)
                std::vector<int32_t> output_ids;
                std::string response;
                std::string interactive_text;
                // Think-block styling: buffer output to suppress <think></think>
                // tags and render thinking content in dim grey.
                std::string print_buf;       // pending text not yet flushed

                // Capture the first token produced during prefill
                // (engine->step() generates it as part of the prefill pass)
                if (ctx->active_request &&
                    !ctx->active_request->output_tokens.empty()) {
                    int32_t first_tok = ctx->active_request->output_tokens.back();
                    output_ids.push_back(first_tok);
                    std::string piece = tok->decode_token(first_tok);
                    interactive_text += piece;
                    print_buf += piece;
                }

                // Decode token by token
                bool in_think = false;
                static const char* kThinkOn  = "\033[2;90m";  // dim + bright black
                static const char* kThinkOff = "\033[0m";

                // Flush confirmed text from print_buf up to a safe point
                auto flush_buf = [&]() {
                    if (print_buf.empty()) return;
                    // Don't flush text that could be a partial tag
                    // Max partial: "</think>" (8 chars) or "<think>" (7 chars)
                    const size_t hold = 8;
                    if (print_buf.size() <= hold) return;
                    size_t safe = print_buf.size() - hold;
                    printf("%.*s", (int)safe, print_buf.c_str());
                    fflush(stdout);
                    print_buf.erase(0, safe);
                };

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
                    interactive_text += piece;
                    print_buf += piece;

                    // Scan for tag transitions in the buffer
                    while (true) {
                        if (!in_think) {
                            auto pos = print_buf.find("<think>");
                            if (pos != std::string::npos) {
                                // Flush text before the tag normally
                                if (pos > 0) {
                                    printf("%.*s", (int)pos, print_buf.c_str());
                                }
                                // Switch to think style, consume the tag
                                printf("%s", kThinkOn);
                                fflush(stdout);
                                print_buf.erase(0, pos + 7);
                                in_think = true;
                                continue;
                            }
                        } else {
                            auto pos = print_buf.find("</think>");
                            if (pos != std::string::npos) {
                                // Flush thinking text before closing tag
                                if (pos > 0) {
                                    printf("%.*s", (int)pos, print_buf.c_str());
                                }
                                // Reset style, consume the tag
                                printf("%s", kThinkOff);
                                fflush(stdout);
                                print_buf.erase(0, pos + 8);
                                in_think = false;
                                continue;
                            }
                        }
                        break;
                    }

                    // Flush safe portion of buffer (keeping potential partial tags)
                    flush_buf();

                    // Check text-level stop sequences
                    if (!args.stop_sequences.empty()) {
                        bool text_stop = false;
                        for (const auto& stop : args.stop_sequences) {
                            if (interactive_text.find(stop) != std::string::npos) {
                                text_stop = true;
                                break;
                            }
                        }
                        if (text_stop) break;
                    }
                }
                // Flush remaining buffer
                if (!print_buf.empty()) {
                    printf("%s", print_buf.c_str());
                }
                if (in_think) printf("%s", kThinkOff);
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
            // Load image if specified
            if (!args.image_path.empty()) {
                err = imp_set_image(ctx, args.image_path.c_str());
                if (err != IMP_SUCCESS) {
                    fprintf(stderr, "Error loading image: %s\n", imp_error_string(err));
                    imp_context_free(ctx);
                    imp_model_free(model);
                    return 1;
                }
                fprintf(stderr, "Image loaded: %s\n", args.image_path.c_str());
            }

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

            // Tokenize prompt (with image tokens if vision is active)
            std::vector<int32_t> tokens;
            if (have_template && ctx->engine->has_vision_input()) {
                std::vector<imp::ChatMessage> msgs = {{"user", args.prompt}};
                tokens = chat_tpl.apply_with_image(*tok, msgs, 256);
            } else if (have_template) {
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

            // Compute max stop length for buffering
            size_t max_stop_len = 0;
            for (const auto& s : args.stop_sequences) max_stop_len = std::max(max_stop_len, s.size());

            // Capture the first token produced during prefill
            auto t_decode_start = std::chrono::high_resolution_clock::now();
            std::vector<int32_t> output_ids;
            std::string output_text;
            if (ctx->active_request &&
                !ctx->active_request->output_tokens.empty()) {
                int32_t first_tok = ctx->active_request->output_tokens.back();
                // Check stop conditions on first token
                bool first_is_stop = (first_tok == tok->eos_id());
                if (!first_is_stop && have_template) {
                    for (int32_t stop_id : chat_tpl.stop_token_ids()) {
                        if (first_tok == stop_id) { first_is_stop = true; break; }
                    }
                }
                if (!first_is_stop) {
                    output_ids.push_back(first_tok);
                    std::string piece = tok->decode_token(first_tok);
                    fprintf(stderr, "[tok=%d '%s'] ", first_tok, piece.c_str());
                    printf("%s", piece.c_str());
                    fflush(stdout);
                    if (!args.stop_sequences.empty()) output_text += piece;
                }
            }

            // Decode remaining tokens
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

                // Check text-level stop sequences
                if (!args.stop_sequences.empty()) {
                    output_text += piece;
                    bool stop_found = false;
                    for (const auto& stop : args.stop_sequences) {
                        if (output_text.find(stop) != std::string::npos) {
                            stop_found = true;
                            break;
                        }
                    }
                    if (stop_found) break;
                }
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
