#include "options.h"

#include "awq_sites.h"
#include "common/exit_codes.h"
#include "tensor_policy.h"
#include "usage.h"

#include "imp/error.h"

#include <cctype>
#include <cstdio>

namespace imp::quantize {

std::optional<int> parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--model")
            opt.in_dir = next();
        else if (a == "--out")
            opt.out_dir = next();
        else if (a == "--lm-head") {
            // Optional value: a bare flag is the pre-#2479 NVFP4 opt-in.
            const bool has_value = i + 1 < argc && argv[i + 1][0] != '-';
            const std::string v = has_value ? next() : "nvfp4";
            if (!parse_lm_head_export(v, opt.lm_head)) {
                fprintf(stderr, "imp-quantize: --lm-head '%s': expected fp8 | nvfp4 | source\n", v.c_str());
                return imp::tools::exit_code_for(IMP_ERROR_INVALID_ARG);
            }
            opt.lm_head_set = true;
        } else if (a == "--kv-hint") {
            const std::string v = next();
            if (!parse_kv_hint(v, opt.kv_hint)) {
                fprintf(stderr, "imp-quantize: --kv-hint '%s': expected auto | fp8 | none\n", v.c_str());
                return imp::tools::exit_code_for(IMP_ERROR_INVALID_ARG);
            }
        } else if (a == "--keep-attn-gate")
            opt.keep_attn_gate = true;
        else if (a == "--keep-gdn-proj") {
            // Optional value: a bare flag keeps all three roles.
            const bool has_value = i + 1 < argc && argv[i + 1][0] != '-';
            opt.keep_gdn_proj = has_value ? next() : "all";
            if (!valid_keep_gdn_proj_selection(opt.keep_gdn_proj)) {
                fprintf(stderr, "imp-quantize: --keep-gdn-proj '%s': expected all or a list of in|gate|out\n",
                        opt.keep_gdn_proj.c_str());
                return imp::tools::exit_code_for(IMP_ERROR_INVALID_ARG);
            }
        } else if (a == "--calib")
            opt.calib_file = next();
        else if (a == "--format") {
            const std::string f = next();
            if (!parse_output_format(f, opt.format)) {
                fprintf(stderr, "imp-quantize: unknown --format '%s' (modelopt | vllm)\n", f.c_str());
                return imp::tools::exit_code_for(IMP_ERROR_INVALID_ARG);
            }
        } else if (a == "--calib-weight") {
            const std::string w = next();
            if (w != "abs" && w != "sq") {
                fprintf(stderr, "imp-quantize: unknown --calib-weight '%s' (abs | sq)\n", w.c_str());
                return imp::tools::exit_code_for(IMP_ERROR_INVALID_ARG);
            }
            opt.calib_weight_sq = (w == "sq");
        } else if (a == "--calib-groups") {
            opt.calib_groups = next();
            for (char& c : opt.calib_groups)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            // An unknown letter here would silently select fewer groups than the
            // caller meant and quietly change what the checkpoint is — the same
            // shape as the `--set` unknown-key hole closed in #1186.
            if (opt.calib_groups.empty() ||
                opt.calib_groups.find_first_not_of(awq::kAwqAllGroups) != std::string::npos) {
                fprintf(stderr, "imp-quantize: --calib-groups '%s' has a letter outside %s\n",
                        opt.calib_groups.c_str(), awq::kAwqAllGroups);
                return imp::tools::exit_code_for(IMP_ERROR_INVALID_ARG);
            }
        } else if (a == "--dry-run")
            opt.dry_run = true;
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return imp::tools::exit_code_for(IMP_ERROR_INVALID_ARG);
        }
    }
    if (opt.in_dir.empty() || (opt.out_dir.empty() && !opt.dry_run)) {
        usage();
        return imp::tools::exit_code_for(IMP_ERROR_INVALID_ARG);
    }
    if (!opt.lm_head_set)
        opt.lm_head = default_lm_head_export(opt.format);
    opt.quantize_lm_head = opt.lm_head == LmHeadExport::Nvfp4;
    return std::nullopt;
}

}  // namespace imp::quantize
