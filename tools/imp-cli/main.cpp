#include "imp/imp.h"
#include "args.h"
#include "model/chat_template.h"
#include "model/tokenizer.h"
#include "runtime/engine.h"

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
    if (args.ssm_fp16) config.ssm_state_dtype = IMP_DTYPE_FP16;

    ImpContext ctx = nullptr;
    err = imp_context_create(model, &config, &ctx);
    if (err != IMP_SUCCESS) {
        fprintf(stderr, "Error creating context: %s\n", imp_error_string(err));
        imp_model_free(model);
        return 1;
    }

    ImpGenerateParams params = imp_generate_params_default();
    params.temperature = args.temperature;
    params.top_p = args.top_p;
    params.top_k = args.top_k;
    params.max_tokens = args.max_tokens;
    params.seed = args.seed;

    // Determine chat template override from --chat-template flag
    if (args.chat_template == "none") {
        params.apply_chat_template = 0;
    }

    if (args.interactive) {
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
        // Single-shot mode: engine handles template via generate()
        if (args.prompt.empty()) {
            fprintf(stderr, "No prompt provided. Use --prompt or --interactive\n");
        } else {
            char output[8192];
            size_t output_len = 0;
            err = imp_generate(ctx, args.prompt.c_str(), &params, output, sizeof(output), &output_len);
            if (err != IMP_SUCCESS) {
                fprintf(stderr, "Generation error: %s\n", imp_error_string(err));
            } else {
                printf("%.*s\n", (int)output_len, output);
            }
        }
    }

    imp_context_free(ctx);
    imp_model_free(model);
    return 0;
}
