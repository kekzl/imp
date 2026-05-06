#include "api/imp_internal.h"
#include "args.h"
#include "model/chat_template.h"
#include "model/hf_hub.h"
#include "model/tokenizer.h"
#include <sys/stat.h>
#include "runtime/presets.h"
#include "runtime/config.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    CliArgs args = parse_args(argc, argv);

    // Load imp.conf (if present) + apply --set overrides; install as the
    // process-wide RuntimeConfig so all downstream code reads through
    // RuntimeConfig::current() instead of getenv("IMP_*").
    imp::RuntimeConfig::install(imp::RuntimeConfig::load(args.config_path, args.config_overrides));

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
        struct stat st;
        if (stat(args.model_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            std::string index = args.model_path + "/model.safetensors.index.json";
            std::string single = args.model_path + "/model.safetensors";
            struct stat idx_st;
            if (stat(index.c_str(), &idx_st) == 0 || stat(single.c_str(), &idx_st) == 0) {
                format = IMP_FORMAT_SAFETENSORS;
            }
        }
    }

    if (format == IMP_FORMAT_GGUF) {
        resolved_model = imp::resolve_model_gguf(args.model_path, args.revision);
        if (resolved_model.empty()) {
            fprintf(stderr, "Failed to resolve model: %s\n", args.model_path.c_str());
            return 1;
        }
        if (resolved_model != args.model_path) {
            printf("Resolved model: %s -> %s\n", args.model_path.c_str(), resolved_model.c_str());
        }
    }

    printf("Loading model: %s (%s)\n", resolved_model.c_str(),
           format == IMP_FORMAT_SAFETENSORS ? "SafeTensors" : "GGUF");

    auto t_init_start = std::chrono::high_resolution_clock::now();

    ImpModel model = nullptr;
    ImpError err = imp_model_load(resolved_model.c_str(), format, &model);
    if (err != IMP_SUCCESS) {
        fprintf(stderr, "Error loading model: %s\n", imp_error_string(err));
        return 1;
    }

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
    if (args.kv_turboquant)
        config.kv_cache_dtype = IMP_DTYPE_TURBOQUANT;
    if (args.kv_turboquant_lite) {
        config.kv_cache_dtype = IMP_DTYPE_TURBOQUANT_LITE;
        config.turboquant_sketch_multiplier = args.turboquant_sketch_mult;
    }
    if (args.ssm_fp16)
        config.ssm_state_dtype = IMP_DTYPE_FP16;
    if (args.no_cuda_graphs)
        config.enable_cuda_graphs = 0;
    if (args.prefill_chunk_size > 0)
        config.prefill_chunk_size = args.prefill_chunk_size;
    if (args.prefill_fp8)
        config.use_fp8_prefill = 1;
    if (args.mxfp4_prefill)
        config.use_mxfp4_prefill = 1;
    if (args.dual_path_quant)
        config.dual_path_quant = 1;
    if (args.prefix_caching)
        config.use_prefix_caching = 1;
    if (args.streaming_kv) {
        config.streaming_kv_enabled = 1;
        config.streaming_kv_n_sinks = args.streaming_sinks;
        config.streaming_kv_window = args.streaming_window;
    }
    config.use_nvfp4_decode = args.decode_nvfp4;
    if (!args.mmproj_path.empty())
        config.mmproj_path = args.mmproj_path.c_str();

    // In bench mode, ensure KV cache matches what the benchmark needs.
    // Raises max_seq_len for long-context benchmarks, caps it for short ones.
    if (args.bench) {
        int bench_need = args.bench_pp + args.max_tokens + 256;  // +256 headroom
        if (config.max_seq_len != bench_need) {
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
        imp_prefill_with_params(ctx, tokens.data(), args.bench_pp, &bench_params);
        for (int s = 0; s < tg_tokens; s++) {
            int32_t tok = 0;
            imp_decode_step(ctx, &bench_params, &tok);
        }

        // PP benchmark
        double pp_total_ms = 0;
        for (int rep = 0; rep < args.bench_reps; rep++) {
            imp_context_reset(ctx);
            auto t0 = std::chrono::high_resolution_clock::now();
            err = imp_prefill_with_params(ctx, tokens.data(), args.bench_pp, &bench_params);
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
            err = imp_prefill_with_params(ctx, tokens.data(), args.bench_pp, &bench_params);
            if (err != IMP_SUCCESS) {
                fprintf(stderr, "Prefill error on tg rep %d: %s\n", rep, imp_error_string(err));
                break;
            }
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < tg_tokens; s++) {
                int32_t tok = 0;
                err = imp_decode_step(ctx, &bench_params, &tok);
                if (err != IMP_SUCCESS)
                    break;
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

        fprintf(stderr, "pp %5d tokens  avg %8.2f ms  (%7.2f tok/s)  [%d reps]\n", args.bench_pp, pp_avg_ms,
                pp_toks, args.bench_reps);
        fprintf(stderr, "tg %5d tokens  avg %8.2f ms  (%7.2f tok/s)  [%d reps]\n", tg_tokens, tg_avg_ms,
                tg_toks, args.bench_reps);
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
            if (!fgets(line, sizeof(line), stdin))
                break;

            // Trim trailing newline
            size_t len = std::strlen(line);
            if (len > 0 && line[len - 1] == '\n')
                line[len - 1] = '\0';

            std::string input(line);
            if (input.empty() || input == "quit" || input == "exit")
                break;

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

                // Prefill with templated tokens (params apply to first sample)
                err = imp_prefill_with_params(ctx, tokens.data(), static_cast<int>(tokens.size()), &params);
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
                std::string print_buf;  // pending text not yet flushed

                // Capture the first token produced during prefill
                // (engine->step() generates it as part of the prefill pass)
                if (ctx->active_request && !ctx->active_request->output_tokens.empty()) {
                    int32_t first_tok = ctx->active_request->output_tokens.back();
                    output_ids.push_back(first_tok);
                    std::string piece = tok->decode_token(first_tok);
                    interactive_text += piece;
                    print_buf += piece;
                }

                // Decode token by token
                bool in_think = false;
                static const char* kThinkOn = "\033[2;90m";  // dim + bright black
                static const char* kThinkOff = "\033[0m";

                // Flush confirmed text from print_buf up to a safe point
                auto flush_buf = [&]() {
                    if (print_buf.empty())
                        return;
                    // Don't flush text that could be a partial tag
                    // Max partial: "</think>" (8 chars) or "<think>" (7 chars)
                    const size_t hold = 8;
                    if (print_buf.size() <= hold)
                        return;
                    size_t safe = print_buf.size() - hold;
                    printf("%.*s", (int)safe, print_buf.c_str());
                    fflush(stdout);
                    print_buf.erase(0, safe);
                };

                for (int step = 0; step < params.max_tokens; step++) {
                    int32_t token = 0;
                    err = imp_decode_step(ctx, &params, &token);
                    if (err != IMP_SUCCESS)
                        break;

                    // Check stop tokens
                    if (token == tok->eos_id())
                        break;
                    bool is_stop = false;
                    for (int32_t stop_id : chat_tpl.stop_token_ids()) {
                        if (token == stop_id) {
                            is_stop = true;
                            break;
                        }
                    }
                    if (is_stop)
                        break;

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
                        if (text_stop)
                            break;
                    }
                }
                // Flush remaining buffer
                if (!print_buf.empty()) {
                    printf("%s", print_buf.c_str());
                }
                if (in_think)
                    printf("%s", kThinkOff);
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
                bool add_bos = tok->add_bos();
                if (imp::RuntimeConfig::current().generation.force_bos)
                    add_bos = true;
                if (add_bos) {
                    tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));
                }
            }
            int n_prompt_tokens = static_cast<int>(tokens.size());
            if (imp::RuntimeConfig::current().diagnostics.dump_tokens) {
                fprintf(stderr, "[DUMP_TOKENS] n=%d:", n_prompt_tokens);
                for (int ti = 0; ti < n_prompt_tokens && ti < 20; ti++)
                    fprintf(stderr, " %d", tokens[ti]);
                fprintf(stderr, "\n");
            }

            // Prefill with timing
            auto t_prefill_start = std::chrono::high_resolution_clock::now();
            err = imp_prefill_with_params(ctx, tokens.data(), n_prompt_tokens, &params);
            auto t_prefill_end = std::chrono::high_resolution_clock::now();
            if (err != IMP_SUCCESS) {
                fprintf(stderr, "Prefill error: %s\n", imp_error_string(err));
                imp_context_free(ctx);
                imp_model_free(model);
                return 1;
            }

            // Compute max stop length for buffering
            size_t max_stop_len = 0;
            for (const auto& s : args.stop_sequences)
                max_stop_len = std::max(max_stop_len, s.size());

            // Capture the first token produced during prefill
            auto t_decode_start = std::chrono::high_resolution_clock::now();
            std::vector<int32_t> output_ids;
            std::string output_text;
            if (ctx->active_request && !ctx->active_request->output_tokens.empty()) {
                int32_t first_tok = ctx->active_request->output_tokens.back();
                // Check stop conditions on first token
                bool first_is_stop = (first_tok == tok->eos_id());
                if (!first_is_stop && have_template) {
                    for (int32_t stop_id : chat_tpl.stop_token_ids()) {
                        if (first_tok == stop_id) {
                            first_is_stop = true;
                            break;
                        }
                    }
                }
                if (!first_is_stop) {
                    output_ids.push_back(first_tok);
                    std::string piece = tok->decode_token(first_tok);
                    fprintf(stderr, "[tok=%d '%s'] ", first_tok, piece.c_str());
                    printf("%s", piece.c_str());
                    fflush(stdout);
                    if (!args.stop_sequences.empty())
                        output_text += piece;
                }
            }

            // Decode remaining tokens
            for (int step = 0; step < params.max_tokens; step++) {
                int32_t token = 0;
                err = imp_decode_step(ctx, &params, &token);
                if (err != IMP_SUCCESS)
                    break;

                // Hide stop tokens from the user but DON'T break the loop —
                // the engine has the authoritative stop logic (think-state
                // suppression, max_tokens budget). When the engine actually
                // finishes the request the next imp_decode_step returns
                // IMP_ERROR_INTERNAL and we exit above. Bailing here on the
                // first eos / im_end stops generation while the engine is
                // still inside a <think> block on Qwen3.6-NVFP4 long-context
                // (model emits <|im_end|> after empty thought; engine flips
                // in_think to false implicitly and continues into the actual
                // answer; CLI was previously cutting it off mid-recovery).
                bool hide_token = (token == tok->eos_id());
                if (have_template && !hide_token) {
                    for (int32_t stop_id : chat_tpl.stop_token_ids()) {
                        if (token == stop_id) {
                            hide_token = true;
                            break;
                        }
                    }
                }

                output_ids.push_back(token);
                std::string piece = tok->decode_token(token);
                if (step < 10)
                    fprintf(stderr, "[tok=%d '%s'] ", token, piece.c_str());
                if (!hide_token) {
                    printf("%s", piece.c_str());
                    fflush(stdout);
                }

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
                    if (stop_found)
                        break;
                }
            }
            auto t_decode_end = std::chrono::high_resolution_clock::now();
            printf("\n");

            int n_output_tokens = static_cast<int>(output_ids.size());
            double prefill_ms =
                std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
            double decode_ms =
                std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
            double total_ms =
                std::chrono::duration<double, std::milli>(t_decode_end - t_prefill_start).count();

            double pp_toks = (prefill_ms > 0) ? (n_prompt_tokens / (prefill_ms / 1000.0)) : 0;
            double tg_toks = (decode_ms > 0 && n_output_tokens > 1)
                                 ? ((n_output_tokens - 1) / (decode_ms / 1000.0))
                                 : 0;

            fprintf(stderr, "\n");
            fprintf(stderr, "pp %5d tokens in %8.2f ms  (%7.2f tok/s)\n", n_prompt_tokens, prefill_ms,
                    pp_toks);
            fprintf(stderr, "tg %5d tokens in %8.2f ms  (%7.2f tok/s)\n", n_output_tokens, decode_ms,
                    tg_toks);
            fprintf(stderr, "total   %8.2f ms\n", total_ms);

            // Benchmark using Engine::generate() (conditional graph loop) for comparison.
            // This eliminates per-step host overhead — shows true GPU-limited throughput.
            if (imp::RuntimeConfig::current().bench.generate) {
                // Reset context for fresh generation
                imp_context_reset(ctx);

                // Use Engine::generate() directly for accurate timing
                imp::Engine* engine = ctx->engine.get();
                auto t_gen_start = std::chrono::high_resolution_clock::now();
                std::string gen_result = engine->generate(args.prompt, params.max_tokens, params.temperature,
                                                          params.top_p, params.top_k, params.seed,
                                                          have_template);
                auto t_gen_end = std::chrono::high_resolution_clock::now();

                // Count output tokens by encoding the result
                auto gen_toks = tok->encode(gen_result);
                int gen_n = static_cast<int>(gen_toks.size());
                double gen_total_ms =
                    std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
                // Estimate decode time: total - prefill (reuse prefill timing from above)
                double gen_decode_ms = gen_total_ms - prefill_ms;
                double gen_toks_s = (gen_decode_ms > 0 && gen_n > 0) ? (gen_n / (gen_decode_ms / 1000.0)) : 0;
                fprintf(stderr, "graph-loop: %d tg tokens in %.2f ms (%.2f tok/s, %.2f ms total)\n", gen_n,
                        gen_decode_ms, gen_toks_s, gen_total_ms);
            }
        }
    }

    imp_context_free(ctx);
    imp_model_free(model);
    return 0;
}
