#include "modes.h"

#include "common/exit_codes.h"
#include "json_report.h"
#include "memory/vram_query.h"

#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <fstream>
#include <iterator>
#include <string>
#include <cstdio>
#include <vector>

namespace imp_cli {

namespace {
// NVTX phase ranges for nsys: tools/roofline/inventory splits kernels by the launching call.
struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};
}  // namespace

int run_bench(ImpContext ctx, ImpModel model, const CliArgs& args, const std::string& resolved_model) {
    ImpError err = IMP_SUCCESS;
    // Synthetic benchmark mode (matches llama-bench methodology)
    int vocab_size = imp_model_vocab_size(model);
    std::vector<int32_t> tokens(args.bench_pp);
    for (int i = 0; i < args.bench_pp; i++)
        tokens[i] = i % vocab_size;
    const int tg_tokens = args.max_tokens;
    // Teacher forcing: tg decode inputs = the file tokens after the prompt (text position pp + k).
    std::vector<int32_t> forced;
    if (args.bench_teacher_force && args.bench_prompt_file.empty()) {
        fprintf(stderr, "Error: --bench-teacher-force needs --bench-prompt-file\n");
        return 1;
    }
    // Real text: the decode continues a prompt the model understands, so the routed experts (and
    // a host-resident MoE's cache hit rate) no longer follow a degenerate continuation of 0..pp-1.
    if (!args.bench_prompt_file.empty()) {
        std::ifstream in(args.bench_prompt_file, std::ios::binary);
        if (!in) {
            fprintf(stderr, "Error: cannot open --bench-prompt-file %s\n", args.bench_prompt_file.c_str());
            return 1;
        }
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const int cap = args.bench_pp + (args.bench_teacher_force ? tg_tokens + 1 : 0);
        std::vector<int32_t> ids(static_cast<size_t>(cap));
        int n_ids = 0;
        err = imp_tokenize(model, text.c_str(), ids.data(), &n_ids, cap);
        if (err != IMP_SUCCESS || n_ids == 0) {
            fprintf(stderr, "Error: --bench-prompt-file %s gave no tokens (%s)\n",
                    args.bench_prompt_file.c_str(), imp_error_string(err));
            return 1;
        }
        for (int i = 0; i < args.bench_pp; i++)
            tokens[i] = ids[i % n_ids];
        fprintf(stderr, "Benchmark prompt: %s, %d tokens%s\n", args.bench_prompt_file.c_str(), n_ids,
                n_ids < args.bench_pp ? " (repeated to pp)" : "");
        if (args.bench_teacher_force) {
            forced.resize(static_cast<size_t>(tg_tokens) + 1);  // [0] replaces the prefill token
            for (int k = 0; k <= tg_tokens; k++)
                forced[k] = ids[(args.bench_pp + k) % n_ids];
        }
    }

    // Greedy decode params for deterministic benchmarking
    ImpGenerateParams bench_params = imp_generate_params_default();
    bench_params.temperature = 0.0f;
    bench_params.ignore_eos = 1;  // Don't stop on EOS during benchmark
    // +1 because imp_prefill already produces the first output token;
    // without this the request hits max_tokens one decode step early.
    bench_params.max_tokens = tg_tokens + 1;
    // Teacher forcing reads its count after the last step: one spare token keeps the request open.
    if (args.bench_teacher_force)
        bench_params.max_tokens += 1;

    fprintf(stderr, "Benchmark: pp=%d, tg=%d, reps=%d\n", args.bench_pp, tg_tokens, args.bench_reps);

    // Warmup: 1 prefill+decode cycle, then prefills until the prefill graph replays. FP8 KV calibrates
    // on prefill 1, the graph arms on 2 and captures on 3, so timed reps start at replay (#2525:
    // a timed eager or capture rep spread pp512 16843..24344 tok/s).
    constexpr int kWarmupPrefills = 3;
    fprintf(stderr, "Warmup... (%d prefills)\n", kWarmupPrefills);
    {
        NvtxRange r("bench:warmup");
        for (int w = 0; w < kWarmupPrefills; w++) {
            if (const ImpError reset_err = imp_context_reset(ctx); reset_err != IMP_SUCCESS) {
                fprintf(stderr, "Context reset error in warmup: %s\n", imp_error_string(reset_err));
                return 1;
            }
            imp_prefill_with_params(ctx, tokens.data(), args.bench_pp, &bench_params);
            if (w == 0 && !forced.empty())  // warm the per-step path the forced tg reps take
                (void)imp_set_forced_decode(ctx, forced.data(), static_cast<int>(forced.size()));
            for (int s = 0; w == 0 && s < tg_tokens; s++) {
                int32_t tok = 0;
                imp_decode_step(ctx, &bench_params, &tok);
            }
        }
    }

    // PP benchmark
    double pp_total_ms = 0;
    for (int rep = 0; rep < args.bench_reps; rep++) {
        if (const ImpError reset_err = imp_context_reset(ctx); reset_err != IMP_SUCCESS) {
            fprintf(stderr, "Context reset error on rep %d: %s\n", rep, imp_error_string(reset_err));
            return 1;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        {
            NvtxRange r("bench:pp");
            err = imp_prefill_with_params(ctx, tokens.data(), args.bench_pp, &bench_params);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        if (err != IMP_SUCCESS) {
            fprintf(stderr, "Prefill error on rep %d: %s\n", rep, imp_error_string(err));
            break;
        }
        const double rep_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "pp rep %d %8.3f ms\n", rep, rep_ms);
        pp_total_ms += rep_ms;
    }

    // TG benchmark
    double tg_total_ms = 0;
    for (int rep = 0; rep < args.bench_reps; rep++) {
        if (const ImpError reset_err = imp_context_reset(ctx); reset_err != IMP_SUCCESS) {
            fprintf(stderr, "Context reset error on tg rep %d: %s\n", rep, imp_error_string(reset_err));
            return 1;
        }
        err = imp_prefill_with_params(ctx, tokens.data(), args.bench_pp, &bench_params);
        if (err != IMP_SUCCESS) {
            fprintf(stderr, "Prefill error on tg rep %d: %s\n", rep, imp_error_string(err));
            break;
        }
        if (!forced.empty())
            err = imp_set_forced_decode(ctx, forced.data(), static_cast<int>(forced.size()));
        if (err != IMP_SUCCESS) {
            fprintf(stderr, "Teacher forcing error on tg rep %d: %s\n", rep, imp_error_string(err));
            break;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        {
            NvtxRange r("bench:tg");
            for (int s = 0; s < tg_tokens; s++) {
                int32_t tok = 0;
                err = imp_decode_step(ctx, &bench_params, &tok);
                if (err != IMP_SUCCESS)
                    break;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        if (err != IMP_SUCCESS) {
            fprintf(stderr, "Decode error on rep %d: %s\n", rep, imp_error_string(err));
            break;
        }
        // Every decode input must have come from the file; a path that bypassed the substitution
        // would decode its own tokens and the rep would not be comparable.
        if (!forced.empty()) {
            const int n_forced = imp_forced_decode_count(ctx);
            if (n_forced != tg_tokens + 1) {
                fprintf(stderr, "Error: teacher forcing covered %d of %d tg inputs on rep %d\n", n_forced - 1,
                        tg_tokens, rep);
                return 1;
            }
            if (rep == 0)
                fprintf(stderr, "Teacher-forced %d of %d tg inputs\n", n_forced - 1, tg_tokens);
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
