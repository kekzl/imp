#include "compute/dispatch_paths.h"

namespace imp {

const char* attn_prefill_path_name(AttnPrefillPath p) {
    switch (p) {
        case AttnPrefillPath::MXFP4:
            return "fmha_mxfp4";
        case AttnPrefillPath::FA2:
            return "fmha_fa2";
        case AttnPrefillPath::FP8:
            return "fmha_fp8";
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
        case AttnDecodePath::NVFP4_TC:
            return "paged_nvfp4_tc";
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
    }
    return "?";
}

const char* moe_prefill_path_name(MoePrefillPath p) {
    switch (p) {
        case MoePrefillPath::DEVICE_ARGS:
            return "device_args";
        case MoePrefillPath::SMALL_M:
            return "small_m";
        case MoePrefillPath::GROUPED:
            return "grouped";
        case MoePrefillPath::LEGACY:
            return "legacy";
    }
    return "?";
}

}  // namespace imp
