#include "modes.h"

#include "common/exit_codes.h"
#include "json_report.h"
#include "memory/vram_query.h"

#include <chrono>
#include <cstdio>
#include <vector>

namespace imp_cli {

int run_bench(ImpContext ctx, ImpModel model, const CliArgs& args, const std::string& resolved_model) {
    ImpError err = IMP_SUCCESS;
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
    fprintf(stderr, "tg %5d tokens  avg %8.2f ms  (%7.2f tok/s)  [%d reps]\n", tg_tokens, tg_avg_ms, tg_toks,
            args.bench_reps);

    if (args.json_out)
        imp_cli::emit_bench({resolved_model, pp_toks, tg_toks, args.bench_pp, pp_avg_ms, tg_tokens, tg_avg_ms,
                             args.bench_reps, static_cast<long long>(imp::vram_own_peak_bytes() >> 20)});
    return 0;
}

}  // namespace imp_cli
