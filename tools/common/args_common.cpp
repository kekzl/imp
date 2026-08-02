#include "args_common.h"

#include <cstdlib>
#include <cstring>

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
    } else if (std::strcmp(a, "--device") == 0 && i + 1 < argc) {
        args.device = std::atoi(argv[++i]);
    } else if (std::strcmp(a, "--gpu-layers") == 0 && i + 1 < argc) {
        args.gpu_layers = std::atoi(argv[++i]);
    } else if (std::strcmp(a, "--max-tokens") == 0 && i + 1 < argc) {
        args.max_tokens = std::atoi(argv[++i]);
    } else if (std::strcmp(a, "--chat-template") == 0 && i + 1 < argc) {
        args.chat_template = argv[++i];
    } else if (std::strcmp(a, "--mmproj") == 0 && i + 1 < argc) {
        args.mmproj_path = argv[++i];
    } else if (std::strcmp(a, "--vram-budget") == 0 && i + 1 < argc) {
        args.vram_budget_mb = std::atoi(argv[++i]);
    } else if (std::strcmp(a, "--min-kv-tokens") == 0 && i + 1 < argc) {
        args.min_kv_tokens = std::atoi(argv[++i]);
    } else if (std::strcmp(a, "--prefill-chunk-size") == 0 && i + 1 < argc) {
        args.prefill_chunk_size = std::atoi(argv[++i]);

        // --- boolean flags ---
    } else if (std::strcmp(a, "--mem-report") == 0) {
        args.mem_report = true;
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
