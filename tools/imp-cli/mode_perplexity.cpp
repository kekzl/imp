#include "modes.h"

#include "json_report.h"

#include <cmath>
#include <cstdio>

namespace imp_cli {

namespace {

// --perplexity-decode <n>: prefill tokens[0..n-1], then force tokens[n..] through the per-step decode
// route (n == 1 rows, is_prefill off) and score each. Scored rows: n .. size-2, the window
// `--perplexity --set diagnostics.ppl_first=<n>` scores on the prefill route.
int run_perplexity_decode(ImpContext ctx, const CliArgs& args, const std::vector<int32_t>& toks,
                          double* out_ppl, int* out_scored) {
    const int n = static_cast<int>(toks.size());
    const int pp = args.perplexity_decode;
    if (pp < 1 || pp > n - 2) {
        fprintf(stderr, "Error: --perplexity-decode %d needs 1 <= n <= tokens - 2 (%d tokens)\n", pp, n);
        return 1;
    }
    const int steps = n - 1 - pp;  // step k consumes toks[pp + k - 1], scores toks[pp + k]
    ImpGenerateParams p = imp_generate_params_default();
    p.temperature = 0.0f;
    p.ignore_eos = 1;
    p.logprobs = 1;
    p.max_tokens = steps + 2;  // prefill's token plus one spare keeps the request open for the readback
    ImpError err = imp_prefill_with_params(ctx, toks.data(), pp, &p);
    if (err == IMP_SUCCESS)
        err = imp_set_forced_decode(ctx, toks.data() + pp, n - pp);
    for (int s = 0; err == IMP_SUCCESS && s < steps; s++) {
        int32_t tok = 0;
        err = imp_decode_step(ctx, &p, &tok);
    }
    if (err != IMP_SUCCESS) {
        fprintf(stderr, "perplexity (decode route) failed: %s\n", imp_error_string(err));
        return 1;
    }
    double nll_sum = 0.0;
    int scored = 0;
    err = imp_forced_decode_nll(ctx, &nll_sum, &scored);
    // Every step must have been forced and scored, or the number is not the decode route's.
    if (err != IMP_SUCCESS || scored != steps || imp_forced_decode_count(ctx) != steps + 1) {
        fprintf(stderr, "Error: decode route scored %d of %d steps (forced %d)\n", scored, steps,
                imp_forced_decode_count(ctx) - 1);
        return 1;
    }
    *out_ppl = std::exp(nll_sum / scored);
    *out_scored = scored;
    return 0;
}

}  // namespace

int run_perplexity(ImpContext ctx, const CliArgs& args, const std::vector<int32_t>& ppl_tokens,
                   const std::string& resolved_model) {
    double ppl = -1.0;
    if (args.perplexity_decode > 0) {
        int scored = 0;
        if (run_perplexity_decode(ctx, args, ppl_tokens, &ppl, &scored) != 0)
            return 1;
        printf("perplexity: %.4f  (%zu tokens, decode route, %d rows scored after %d prefilled)\n", ppl,
               ppl_tokens.size(), scored, args.perplexity_decode);
        if (args.json_out)
            emit_perplexity(resolved_model, ppl, static_cast<long long>(ppl_tokens.size()),
                            args.perplexity_file, args.calibrate_out);
        return 0;
    }
    const ImpError pe = imp_perplexity(ctx, ppl_tokens.data(), static_cast<int>(ppl_tokens.size()), &ppl);
    if (pe != IMP_SUCCESS) {
        fprintf(stderr, "perplexity failed: %s\n", imp_error_string(pe));
        return 1;
    }
    printf("perplexity: %.4f  (%zu tokens)\n", ppl, ppl_tokens.size());
    if (!args.calibrate_out.empty()) {
        const ImpError ce = imp_calibration_write(ctx, args.calibrate_out.c_str());
        if (ce != IMP_SUCCESS) {
            fprintf(stderr, "calibration write failed: %s\n", imp_error_string(ce));
            return 1;
        }
        printf("calibration: %s\n", args.calibrate_out.c_str());
    }
    if (args.json_out)
        emit_perplexity(resolved_model, ppl, static_cast<long long>(ppl_tokens.size()), args.perplexity_file,
                        args.calibrate_out);
    return 0;
}

}  // namespace imp_cli
