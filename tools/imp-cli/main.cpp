#include "imp/imp.h"
#include "args.h"

#include <cstdio>
#include <cstdlib>
#include <string>

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
    config.max_seq_len = 4096;

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

    if (args.interactive) {
        printf("Interactive mode. Type 'quit' to exit.\n");
        char line[4096];
        while (true) {
            printf("\n> ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            std::string input(line);
            if (input.empty() || input == "quit\n" || input == "exit\n") break;

            char output[8192];
            size_t output_len = 0;
            err = imp_generate(ctx, input.c_str(), &params, output, sizeof(output), &output_len);
            if (err != IMP_SUCCESS) {
                fprintf(stderr, "Generation error: %s\n", imp_error_string(err));
                continue;
            }
            printf("%.*s\n", (int)output_len, output);
            imp_context_reset(ctx);
        }
    } else {
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
