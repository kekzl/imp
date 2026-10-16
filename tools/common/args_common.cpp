#include "args_common.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

bool parse_common_flag(CommonArgs& args, int argc, char** argv, int& i) {
    const char* a = argv[i];

    // --- flags taking a value ---
    if (std::strcmp(a, "--model") == 0 && i + 1 < argc) {
        args.model_path = argv[++i];
    } else if (std::strcmp(a, "--revision") == 0 && i + 1 < argc) {
        args.revision = argv[++i];
    } else if (std::strcmp(a, "--config") == 0 && i + 1 < argc) {
        args.config_path = argv[++i];
    } else if (std::strcmp(a, "--set") == 0 && i + 1 < argc) {
        args.config_overrides.push_back(argv[++i]);
    } else if (std::strcmp(a, "--gpu-layers") == 0 && i + 1 < argc) {
        args.gpu_layers = cli_int(a, argv[++i]);
    } else if (std::strcmp(a, "--max-tokens") == 0 && i + 1 < argc) {
        args.max_tokens = cli_int(a, argv[++i]);
    } else if (std::strcmp(a, "--chat-template") == 0 && i + 1 < argc) {
        args.chat_template = argv[++i];
    } else if (std::strcmp(a, "--mmproj") == 0 && i + 1 < argc) {
        args.mmproj_path = argv[++i];
    } else if (std::strcmp(a, "--vram-budget") == 0 && i + 1 < argc) {
        args.vram_budget_mb = cli_int(a, argv[++i]);
    } else if (std::strcmp(a, "--min-kv-tokens") == 0 && i + 1 < argc) {
        args.min_kv_tokens = cli_int(a, argv[++i]);
    } else if (std::strcmp(a, "--prefill-chunk-size") == 0 && i + 1 < argc) {
        args.prefill_chunk_size = cli_int(a, argv[++i]);

        // --- boolean flags ---
    } else if (std::strcmp(a, "--json") == 0) {
        args.json_out = true;
    } else if (std::strcmp(a, "--no-cuda-graphs") == 0) {
        args.no_cuda_graphs = true;
    } else if (std::strcmp(a, "--kv-fp8") == 0) {
        args.kv_fp8 = true;
    } else if (std::strcmp(a, "--kv-int8") == 0) {
        args.kv_int8 = true;
    } else if (std::strcmp(a, "--kv-int4") == 0) {
        args.kv_int4 = true;
    } else if (std::strcmp(a, "--kv-nvfp4") == 0) {
        args.kv_nvfp4 = true;
    } else if (std::strcmp(a, "--kv-mxfp4") == 0) {
        args.kv_mxfp4 = true;
    } else if (std::strcmp(a, "--ssm-fp16") == 0) {
        args.ssm_fp16 = true;
    } else if (std::strcmp(a, "--decode-nvfp4") == 0) {
        args.decode_nvfp4 = 1;
    } else if (std::strcmp(a, "--decode-nvfp4-only") == 0) {
        args.decode_nvfp4 = 2;
    } else if (std::strcmp(a, "--no-nvfp4") == 0) {
        args.decode_nvfp4 = 0;
    } else if (std::strcmp(a, "--mxfp4-prefill") == 0) {
        args.mxfp4_prefill = true;
    } else if (std::strcmp(a, "--dual-path-quant") == 0) {
        args.dual_path_quant = true;
    } else {
        return false;  // not ours — the caller's own chain gets it
    }
    return true;
}

bool parse_cli_int(const char* s, int& out) {
    if (s == nullptr || *s == '\0')
        return false;
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max())
        return false;
    out = static_cast<int>(v);
    return true;
}

bool parse_cli_double(const char* s, double& out) {
    if (s == nullptr || *s == '\0')
        return false;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || errno == ERANGE || !std::isfinite(v))
        return false;
    out = v;
    return true;
}

int cli_int(const char* flag, const char* value) {
    int v = 0;
    if (!parse_cli_int(value, v)) {
        fprintf(stderr, "%s expects an integer, got '%s'\n", flag, value);
        std::exit(1);
    }
    return v;
}

double cli_double(const char* flag, const char* value) {
    double v = 0.0;
    if (!parse_cli_double(value, v)) {
        fprintf(stderr, "%s expects a number, got '%s'\n", flag, value);
        std::exit(1);
    }
    return v;
}

float cli_float(const char* flag, const char* value) {
    const double v = cli_double(flag, value);
    if (std::fabs(v) > std::numeric_limits<float>::max()) {
        fprintf(stderr, "%s expects a number, got '%s'\n", flag, value);
        std::exit(1);
    }
    return static_cast<float>(v);
}

std::string resolve_calibration_out(const std::string& flag_value, const std::string& config_value) {
    if (!flag_value.empty())
        return flag_value;
    return config_value;
}
