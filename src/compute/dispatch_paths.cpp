#include "compute/dispatch_paths.h"
#include "compute/dispatch_record.h"

#include <bit>
#include <cstddef>
#include <iterator>

namespace imp {

const char* attn_prefill_path_name(AttnPrefillPath p) {
    switch (p) {
        case AttnPrefillPath::FA2:
            return "fmha_fa2";
        case AttnPrefillPath::FMHA_SM120:
            return "fmha_sm120_wmma";
        case AttnPrefillPath::BLACKWELL:
            return "flash_blackwell";
        case AttnPrefillPath::NONE:
            return "none";
    }
    return "?";
}

const char* attn_prefill_outer_name(AttnPrefillOuter p) {
    switch (p) {
        case AttnPrefillOuter::UNSET:
            return "unset";
        case AttnPrefillOuter::FA2_FP16QK:
            return "fa2_fp16qk";
        case AttnPrefillOuter::CUBLAS:
            return "cublas_materialized";
        case AttnPrefillOuter::CUBLAS_SLICED:
            return "cublas_sliced";
        case AttnPrefillOuter::FMHA_CHAIN:
            return "fmha_chain";
        case AttnPrefillOuter::RA2:
            return "ra2_fp4";
        case AttnPrefillOuter::APA:
            return "apa";
    }
    return "?";
}

const char* attn_decode_path_name(AttnDecodePath p) {
    switch (p) {
        case AttnDecodePath::UNSET:
            return "unset";
        case AttnDecodePath::FP16:
            return "paged_fp16";
        case AttnDecodePath::FP8:
            return "paged_fp8";
        case AttnDecodePath::INT8:
            return "paged_int8";
        case AttnDecodePath::INT4:
            return "paged_int4";
        case AttnDecodePath::NVFP4:
            return "paged_nvfp4";
        case AttnDecodePath::MXFP4_KV:
            return "paged_mxfp4_kv";
    }
    return "?";
}

const char* moe_prefill_outer_name(MoePrefillOuter p) {
    switch (p) {
        case MoePrefillOuter::UNSET:
            return "unset";
        case MoePrefillOuter::NONE:
            return "n/a (dense)";
        case MoePrefillOuter::FP16_BATCH:
            return "fp16_batch";
        case MoePrefillOuter::FP8_BATCH:
            return "fp8_batch";
        case MoePrefillOuter::CUTLASS3X:
            return "cutlass3x";
        case MoePrefillOuter::NVFP4_DEQUANT:
            return "nvfp4_dequant_batch";
        case MoePrefillOuter::LEGACY:
            return "legacy_fallback";
        case MoePrefillOuter::FUSED_Q6K:
            return "fused_q6k";
        case MoePrefillOuter::FUSED_Q4K_DP4A:
            return "fused_q4k_dp4a";
    }
    return "?";
}

const char* moe_prefill_path_name(MoePrefillPath p) {
    switch (p) {
        case MoePrefillPath::DEVICE_ARGS:
            return "device_args";
        case MoePrefillPath::GROUPED:
            return "grouped";
        case MoePrefillPath::LEGACY:
            return "legacy";
    }
    return "?";
}

// A table, not a switch (CCN ratchet): one name per DensePath in enum order.
const char* dense_path_name(DensePath p) {
    static constexpr const char* kNames[] = {
        "unset",
        "mxfp8_prefill_copy",
        "nvfp4_prefill_copy",
        "released_source",
        "source_gemv",
        "uncached_fallback",
        "gemv_nvfp4",
        "gemv_fp8",
        "gemv_fp8_rowscale",
        "gemv_nvfp4_source",
        "gemv_dispatch",
        "verify_nvfp4_overlay",
        "verify_nvfp4_source",
        "smallm_a16",
        "smallm_a4",
        "gguf_in_place",
        "dp4a_q4k",
        "dp4a_q5k",
        "fp16_cache",
        "fp8_prefill",
        "imma_source",
        "gguf_source_dequant",
        "nvfp4_prompt_a16",
        "nvfp4_tile",
        "nvfp4_tier_fp8",
        "nvfp4_dequant",
        "mxfp4_prefill",
        "gguf_source_no_sfatom",
        "generic_dispatch",
        "qkv_gemv_mxfp4",
        "qkv_gemv_nvfp4",
        "qkv_dp4a",
        "dp4a_multi_m1",
        "dp4a_gemv",
        "dp4a_residual",
        "gate_up_gemv_mxfp4",
        "gate_up_gemv_nvfp4",
        "gate_up_dp4a",
        "fp16_pair_batched",
        "smallm_multi_ws",
        "smallm_multi_v2",
        "fp8_smallm_multi",
        "fp8_smallm",
        "fp8_tile",
        "fp8_cublaslt",
        "nvfp4_tile_swiglu_quant",
        "nvfp4_prequant",
        "gemv_mxfp4_residual",
        "gemv_nvfp4_residual",
        "gemv_fp8_residual",
        "gdn_input_gemv_nvfp4",
        "gdn_input_gemv_fp8",
        "f16_narrow",
    };
    static_assert(std::size(kNames) == static_cast<size_t>(DensePath::COUNT), "one name per DensePath");
    const auto i = static_cast<size_t>(p);
    return i < std::size(kNames) ? kNames[i] : "?";
}

namespace dispatch_record {

std::string dense_summary(const Record& r) {
    std::string out;
    for (int k = 0; k < kDenseKinds; ++k) {
        const uint64_t seen = r.dense_seen[k];
        if (seen == 0)
            continue;
        if (!out.empty())
            out += ' ';
        out += tensor_kind_name(static_cast<TensorKind>(k));
        char sep = '=';
        for (int p = 0; p < static_cast<int>(DensePath::COUNT); ++p) {
            if ((seen >> p) & 1u) {
                out += sep;
                out += dense_path_name(static_cast<DensePath>(p));
                sep = '+';
            }
        }
    }
    return out;
}

std::string dense_log_line(Record& r, bool first) {
    int fresh = 0;
    for (int k = 0; k < kDenseKinds; ++k) {
        fresh += std::popcount(r.dense_seen[k] & ~r.dense_logged[k]);
        r.dense_logged[k] = r.dense_seen[k];
    }
    if (!first && fresh == 0)
        return {};
    const std::string summary = dense_summary(r);
    if (first)
        return "Resolved dense GEMM: " + (summary.empty() ? std::string("none recorded") : summary);
    return "Resolved dense GEMM (+" + std::to_string(fresh) + " new): " + summary;
}

}  // namespace dispatch_record

}  // namespace imp
