#include "modes.h"

#include "common/exit_codes.h"
#include "json_report.h"
#include "model/chat_template.h"
#include "model/image_placeholders.h"
#include "model/tokenizer.h"
#include "runtime/engine.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace imp_cli {

int run_oneshot(ImpContext ctx, ImpModel model, const CliArgs& args, ImpGenerateParams params,
                const std::string& resolved_model) {
    ImpError err = IMP_SUCCESS;
    // Single-shot mode with timing
    if (args.prompt.empty()) {
        fprintf(stderr, "No prompt provided. Use --prompt, --prompt-file or --interactive\n");
    } else {
        // Load images if specified. The first replaces whatever was
        // pending, the rest append, so repeating --image builds a prompt
        // with one placeholder per picture.
        for (size_t i = 0; i < args.image_paths.size(); ++i) {
            const char* p = args.image_paths[i].c_str();
            err = (i == 0) ? imp_set_image(ctx, p) : imp_add_image(ctx, p);
            if (err != IMP_SUCCESS) {
                fprintf(stderr, "Error loading image '%s': %s\n", p, imp_error_string(err));
                imp_context_free(ctx);
                imp_model_free(model);
                return imp::tools::exit_code_for(err);
            }
            fprintf(stderr, "Image loaded: %s\n", p);
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
        const int pending_img_tokens = imp_pending_image_tokens(ctx);
        if (have_template && pending_img_tokens > 0) {
            // Dynamic resolution: the template emits one <|image_pad|> per
            // block, because the count is not knowable until the image has
            // been resized. Render one block per image, then expand each to
            // what its own encoder pass produced.
            const std::vector<int> counts = ctx->engine->pending_image_token_counts();
            std::string blocks;
            for (size_t i = 0; i < counts.size(); ++i)
                blocks += "<|vision_start|><|image_pad|><|vision_end|>";
            std::vector<imp::ChatMessage> msgs = {{"user", blocks + args.prompt}};
            tokens = chat_tpl.apply(*tok, msgs);
            const int32_t pad_id = tok->find_token("<|image_pad|>");
            const auto expanded = pad_id < 0 ? std::unexpected(std::string("tokenizer has no <|image_pad|>"))
                                             : imp::expand_image_placeholders(tokens, pad_id, counts);
            if (!expanded) {
                fprintf(stderr, "Error placing image tokens: %s\n", expanded.error().c_str());
                imp_context_free(ctx);
                imp_model_free(model);
                return 1;
            }
        } else if (have_template && ctx->engine->has_vision_input()) {
            std::vector<imp::ChatMessage> msgs = {{"user", args.prompt}};
            tokens = chat_tpl.apply_with_image(*tok, msgs, 256);
        } else if (have_template) {
            std::vector<imp::ChatMessage> msgs = {{"user", args.prompt}};
            tokens = chat_tpl.apply(*tok, msgs);
        } else {
            tokens = tok->encode(args.prompt);
            // Prepend BOS when the tokenizer requires it (e.g. Gemma)
            bool add_bos = tok->add_bos();
            if (ctx->engine->runtime_config().generation.force_bos)
                add_bos = true;
            if (add_bos) {
                tokens.insert(tokens.begin(), static_cast<int32_t>(tok->bos_id()));
            }
        }
        int n_prompt_tokens = static_cast<int>(tokens.size());
        if (ctx->engine->runtime_config().diagnostics.dump_tokens) {
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
            return imp::tools::exit_code_for(err);
        }

        // Compute max stop length for buffering
        size_t max_stop_len = 0;
        for (const auto& s : args.stop_sequences)
            max_stop_len = std::max(max_stop_len, s.size());

        // Resolve think token IDs for output filtering.
        // find_token("<think>") fails on Qwen3 BPE where <think> is a
        // regular vocab token (ID 123649, decodes to bytes not "<think>").
        // Fall back to encode() which handles both special and BPE tokens.
        int32_t think_start = tok->find_token("<think>");
        int32_t think_end = tok->find_token("</think>");
        if (think_start < 0) {
            auto ids = tok->encode("<think>");
            if (ids.size() == 1)
                think_start = ids[0];
        }
        if (think_end < 0) {
            auto ids = tok->encode("</think>");
            if (ids.size() == 1)
                think_end = ids[0];
        }
        bool in_think = false;

        auto t_decode_start = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> output_ids;
        std::string output_text;
        // Exactly what stdout shows, for --json (#1583). Not
        // tok->decode(output_ids): that one carries the hidden stop and
        // think markers, so the document would disagree with the terminal.
        std::string visible_text;
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
                if (think_start >= 0 && first_tok == think_start) {
                    in_think = true;
                } else if (think_end >= 0 && first_tok == think_end) {
                    in_think = false;
                } else if (!in_think) {
                    std::string piece = tok->decode_token(first_tok);
                    fprintf(stderr, "[tok=%d '%s'] ", first_tok, piece.c_str());
                    printf("%s", piece.c_str());
                    visible_text += piece;
                    fflush(stdout);
                    if (!args.stop_sequences.empty())
                        output_text += piece;
                }
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
            if (think_start >= 0 && token == think_start) {
                in_think = true;
                hide_token = true;
            } else if (think_end >= 0 && token == think_end) {
                in_think = false;
                hide_token = true;
            } else if (in_think) {
                hide_token = true;
            }
            std::string piece = tok->decode_token(token);
            if (step < 10)
                fprintf(stderr, "[tok=%d '%s'] ", token, piece.c_str());
            if (!hide_token) {
                printf("%s", piece.c_str());
                visible_text += piece;
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
        double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
        double total_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_prefill_start).count();

        double pp_toks = (prefill_ms > 0) ? (n_prompt_tokens / (prefill_ms / 1000.0)) : 0;
        double tg_toks = (decode_ms > 0 && n_output_tokens > 1)
                             ? ((n_output_tokens - 1) / (decode_ms / 1000.0))
                             : 0;

        fprintf(stderr, "\n");
        fprintf(stderr, "pp %5d tokens in %8.2f ms  (%7.2f tok/s)\n", n_prompt_tokens, prefill_ms, pp_toks);
        fprintf(stderr, "tg %5d tokens in %8.2f ms  (%7.2f tok/s)\n", n_output_tokens, decode_ms, tg_toks);
        fprintf(stderr, "total   %8.2f ms\n", total_ms);

        // The generated text is on stderr with everything else under
        // --json, so it has to come back in the document or the mode
        // would silently swallow the answer.
        if (args.json_out)
            imp_cli::emit_generate({resolved_model, visible_text, n_prompt_tokens, n_output_tokens, pp_toks,
                                    tg_toks, prefill_ms, decode_ms, total_ms});

        // Phase 3.5 telemetry: report MTP draft accuracy if measured.
        if (imp::Engine* engine = ctx->engine.get(); engine && engine->mtp_spec_decode_enabled()) {
            auto acc = engine->mtp_accuracy();
            if (acc.total > 0) {
                fprintf(stderr, "mtp     %d / %d drafts matched (%.1f%% accept rate)\n", acc.matches,
                        acc.total, 100.0f * acc.rate());
            }
            // Per-lookahead chain accept (K>1 measurement). chain_accept_[0]
            // duplicates mtp accuracy above; print [1..] only when present.
            auto chain = engine->mtp_chain_accept();
            for (size_t k = 1; k < chain.size(); ++k) {
                if (chain[k].total > 0) {
                    fprintf(stderr, "mtp[k=%zu] %d / %d drafts matched (%.1f%% accept @ +%zu lookahead)\n", k,
                            chain[k].matches, chain[k].total, 100.0f * chain[k].rate(), k);
                }
            }

            // Stage 0 tree-ceiling table: per-depth top-w hit rate.
            // depth d=k+1; width w=1..W. lookahead-0 (depth 1) is
            // teacher-forced; depth ≥2 is self-chained (lower bound).
            auto cw = engine->mtp_chain_accept_width();
            constexpr int W = imp::Engine::kMtpMeasureW;
            if (!cw.empty()) {
                fprintf(stderr, "\n-- MTP tree-ceiling probe (top-w hit rate per depth) --\n");
                fprintf(stderr, "depth   n  ");
                for (int w = 0; w < W; ++w)
                    fprintf(stderr, "  top-%d", w + 1);
                fprintf(stderr, "   (depth1=teacher-forced, depth>=2=self-chained lower bound)\n");
                for (size_t k = 0; k < cw.size(); ++k) {
                    if (cw[k].total == 0)
                        continue;
                    fprintf(stderr, "  %2zu %5d  ", k + 1, cw[k].total);
                    for (int w = 0; w < W; ++w)
                        fprintf(stderr, " %5.1f%%", 100.0f * cw[k].rate(w));
                    fprintf(stderr, "\n");
                }
                // Derived expected accept length (tokens emitted per verify):
                // E[accept] = sum_{d>=1} prod_{j=1..d} p(j), with the bonus
                // token = +1. Linear uses top-1; tree uses top-w per depth.
                for (int w = 0; w < W; ++w) {
                    double e = 0.0, prod = 1.0;
                    for (size_t k = 0; k < cw.size(); ++k) {
                        if (cw[k].total == 0)
                            break;
                        prod *= cw[k].rate(w);
                        e += prod;
                    }
                    fprintf(stderr, "  E[accept] top-%d = %.3f draft tokens%s\n", w + 1, e,
                            w == 0 ? " (linear baseline)" : "");
                }
            }
        }

        // Benchmark using Engine::generate() (conditional graph loop) for comparison.
        // This eliminates per-step host overhead — shows true GPU-limited throughput.
        if (ctx->engine->runtime_config().bench.generate) {
            // Reset context for fresh generation
            imp_context_reset(ctx);

            // Use Engine::generate() directly for accurate timing
            imp::Engine* engine = ctx->engine.get();
            auto t_gen_start = std::chrono::high_resolution_clock::now();
            std::string gen_result = engine->generate(args.prompt, params.max_tokens, params.temperature,
                                                      params.top_p, params.top_k, params.seed, have_template);
            auto t_gen_end = std::chrono::high_resolution_clock::now();

            // Count output tokens by encoding the result
            auto gen_toks = tok->encode(gen_result);
            int gen_n = static_cast<int>(gen_toks.size());
            double gen_total_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
            // Estimate decode time: total - prefill (reuse prefill timing from above)
            double gen_decode_ms = gen_total_ms - prefill_ms;
            double gen_toks_s = (gen_decode_ms > 0 && gen_n > 0) ? (gen_n / (gen_decode_ms / 1000.0)) : 0;
            fprintf(stderr, "graph-loop: %d tg tokens in %.2f ms (%.2f tok/s, %.2f ms total)\n", gen_n,
                    gen_decode_ms, gen_toks_s, gen_total_ms);
        }
    }
    return 0;
}

}  // namespace imp_cli
