#pragma once

// The identities of the kernel paths imp can resolve to, as plain enums.
//
// Deliberately free of RuntimeConfig, CUDA and Tensor so that BOTH consumers
// can include it without dragging a layer with them:
//   - the pure routing models (attention_dispatch_decision.h,
//     exec/moe_prefill_decision.h), which pull in runtime/config.h,
//   - the dispatch recorder (dispatch_record.h), which is included from the
//     hot path and must stay weightless.
//
// Splitting the enums out is what keeps the recorded path and the modelled
// path expressed in ONE vocabulary: a tier renamed here breaks both sides.

namespace imp {

// Tiers inside the FMHA chain (compute/attention_dispatch.cu), tried in order.
enum class AttnPrefillPath {
    MXFP4,       // fmha_sm120_mxfp4_prefill
    FA2,         // fmha_sm120_fa2_prefill (register-resident)
    FP8,         // fmha_sm120_fp8_prefill
    FMHA_SM120,  // fmha_sm120_prefill (WMMA)
    BLACKWELL,   // flash_attention_blackwell (final tier; declines unsupported configs)
    NONE,        // chain exhausted → attention_prefill_dispatch throws (#654)
};

// The branch taken by the executor BEFORE the FMHA chain is even reached
// (exec/executor_attention_prefill.cu). Only FMHA_CHAIN continues into
// AttnPrefillPath above; the other three are terminal.
enum class AttnPrefillOuter {
    UNSET,
    FA2_FP16QK,     // try_fa2_fp16qk_prefill — the primary hd=128/256 path
    CUBLAS,         // attention_cublas_prefill — materialized S-matrix
    CUBLAS_SLICED,  // attention_cublas_prefill_sliced — hd=512 S-matrix overflow (#1036)
    FMHA_CHAIN,     // attention_prefill_dispatch → AttnPrefillPath
};

// Paged-decode kernel family, selected by the KV cache dtype
// (exec/executor_attention_decode.cu).
enum class AttnDecodePath {
    UNSET,
    FP16,
    FP8,
    INT8,
    INT4,
    NVFP4,
    NVFP4_TC,
    MXFP4_KV,
};

// MoE prefill: the outer 5-way chain in exec/executor_forward_moe.cu.
enum class MoePrefillOuter {
    UNSET,
    NONE,           // dense model — no MoE layers
    FP16_BATCH,     // try_run_moe_fp16_batch_prefill
    FP8_BATCH,      // try_run_moe_fp8_batch_prefill
    CUTLASS3X,      // try_run_moe_cutlass3x_nvfp4_prefill_ → MoePrefillPath
    NVFP4_DEQUANT,  // try_run_moe_nvfp4_dequant_batch_prefill_
    LEGACY,         // run_moe_legacy_fallback_
    FUSED_Q6K,      // fused Q6_K path (taken before the chain)
};

// Tiers inside try_run_moe_cutlass3x_nvfp4_prefill_
// (exec/executor_forward_moe_cutlass.cu).
enum class MoePrefillPath {
    DEVICE_ARGS,  // CUTLASS 3.x NVFP4 device-args full path (fast)
    SMALL_M,      // gemm_grouped_nvfp4_smallM (small token counts)
    GROUPED,      // host-args CUTLASS 3.x grouped GEMM (+ gpt-oss bias seams)
    LEGACY,       // per-expert legacy fallback (gather + serial GEMM)
};

const char* attn_prefill_path_name(AttnPrefillPath p);
const char* attn_prefill_outer_name(AttnPrefillOuter p);
const char* attn_decode_path_name(AttnDecodePath p);
const char* moe_prefill_outer_name(MoePrefillOuter p);
const char* moe_prefill_path_name(MoePrefillPath p);

}  // namespace imp
