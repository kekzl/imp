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

int run_interactive(ImpContext ctx, ImpModel model, const CliArgs& args, ImpGenerateParams params) {
    ImpError err = IMP_SUCCESS;
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
            // Repeating /image before sending stacks the pictures up. The
            // pending list empties when a message consumes it, so the first
            // /image of the next turn starts fresh without a clear command.
            const bool first = imp_pending_image_tokens(ctx) == 0;
            err = first ? imp_set_image(ctx, img_path.c_str()) : imp_add_image(ctx, img_path.c_str());
            if (err != IMP_SUCCESS) {
                fprintf(stderr, "Error loading image: %s\n", imp_error_string(err));
            } else {
                printf("Image loaded: %s (applies to your next message)\n", img_path.c_str());
            }
            continue;
        }

        if (have_template) {
            // Multi-turn: append user message and apply full template
            history.push_back({"user", input});
            std::vector<int32_t> tokens;
            // Same three-way split as the single-prompt path below: a
            // pending Qwen-VL image has a count only the encoder knows, an
            // mmproj image has a fixed one, and most turns have neither.
            // Testing only `has_vision_input()` here — which is the mmproj
            // tower alone — rendered a prompt with no image tokens at all,
            // so a picture loaded with /image was never referenced and the
            // model answered as if it had not been given one.
            const int pending_img_tokens = imp_pending_image_tokens(ctx);
            if (pending_img_tokens > 0) {
                const std::vector<int> counts = ctx->engine->pending_image_token_counts();
                std::string blocks;
                for (size_t i = 0; i < counts.size(); ++i)
                    blocks += "<|vision_start|><|image_pad|><|vision_end|>";
                std::vector<imp::ChatMessage> msgs = history;
                msgs.back().content = blocks + msgs.back().content;
                tokens = chat_tpl.apply(*tok, msgs);
                const int32_t pad_id = tok->find_token("<|image_pad|>");
                const auto expanded = pad_id < 0
                                          ? std::unexpected(std::string("tokenizer has no <|image_pad|>"))
                                          : imp::expand_image_placeholders(tokens, pad_id, counts);
                if (!expanded) {
                    fprintf(stderr, "Error placing image tokens: %s\n", expanded.error().c_str());
                    history.pop_back();
                    continue;
                }
            } else if (ctx->engine->has_vision_input()) {
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
    return 0;
}

}  // namespace imp_cli
