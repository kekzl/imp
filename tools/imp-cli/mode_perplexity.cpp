#include "modes.h"

#include "json_report.h"

#include <cstdio>

namespace imp_cli {

int run_perplexity(ImpContext ctx, const CliArgs& args, const std::vector<int32_t>& ppl_tokens,
                   const std::string& resolved_model) {
    double ppl = -1.0;
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
