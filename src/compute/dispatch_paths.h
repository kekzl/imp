#pragma once

// Identities of the kernel paths imp can resolve to, as plain enums. Deliberately free of
// RuntimeConfig/CUDA/Tensor so both the pure routing models (attention_dispatch_decision.h,
// moe_prefill_decision.h) and the hot-path dispatch recorder (dispatch_record.h) can include it
// without dragging in a heavier layer. One vocabulary keeps a renamed tier consistent on both sides.

namespace imp {

// Tiers inside the FMHA chain (compute/attention_dispatch.cpp), tried in order.
enum class AttnPrefillPath {
    FA2,         // fmha_sm120_fa2_prefill (register-resident)
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
    RA2,            // try_ra2_prefill: FP4 prefill, opt-in attention.ra2_prefill
    APA,            // try_apa_prefill: two-pass adaptive precision, hd 128, kv >= attention.apa_min_kv
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
    MXFP4_KV,
};

// MoE prefill: the outer 5-way chain in exec/executor_forward_moe.cu.
enum class MoePrefillOuter {
    UNSET,
    NONE,            // dense model — no MoE layers
    FP16_BATCH,      // try_run_moe_fp16_batch_prefill
    FP8_BATCH,       // try_run_moe_fp8_batch_prefill
    CUTLASS3X,       // try_run_moe_cutlass3x_nvfp4_prefill_ → MoePrefillPath (own grouped GEMM)
    NVFP4_DEQUANT,   // try_run_moe_nvfp4_dequant_batch_prefill_
    LEGACY,          // run_moe_legacy_fallback_
    FUSED_Q6K,       // try_run_moe_q6k_prefill (gemm.moe_imma_prefill=false only)
    FUSED_Q4K_DP4A,  // try_run_moe_q4k_prefill (gemm.moe_imma_prefill=false, eff <= 640)
};

// Tiers inside try_run_moe_cutlass3x_nvfp4_prefill_
// (exec/executor_forward_moe_cutlass.cpp).
enum class MoePrefillPath {
    DEVICE_ARGS,  // gemm_grouped_nvfp4_dispatch, device-args full path (fast)
    GROUPED,      // gemm_grouped_nvfp4_host_args (+ gpt-oss bias seams)
    LEGACY,       // per-expert legacy fallback (gather + serial GEMM)
};

// Exits of the dense GEMM ladder GraphExecutor::gemm_via_handle_ (exec/executor_gemm_dispatch.cpp),
// in ladder order. The ladder returns one per exit, so a new rung must name its path to compile.
// After GENERIC_DISPATCH: the fused / sibling projection routes that bypass the ladder.
enum class DensePath : unsigned char {
    UNSET,
    MXFP8_PREFILL_COPY,     // MXFP8 copy of an F16 GDN projection, M > 32
    NVFP4_PREFILL_COPY,     // NVFP4 copy of an F16 GDN projection, M > 32
    RELEASED_SOURCE,        // released_source_gemm_ (Phase 4b dropped the source)
    SOURCE_GEMV,            // source_gemv: M=1 on the quantized source (dp4a / MMVQ)
    UNCACHED_FALLBACK,      // gemm_dispatch_uncached_fallback: no tier entry, the source is the weight
    GEMV_NVFP4,             // M=1 gemv_nvfp4_kpar on the NVFP4 cache entry
    GEMV_FP8,               // M=1 gemv_fp8, per-tensor scale
    GEMV_FP8_ROWSCALE,      // M=1 gemv_fp8_rowscale
    GEMV_NVFP4_SOURCE,      // M=1 gemv_nvfp4_kpar on the native NVFP4 source
    GEMV_DISPATCH,          // M=1 gemv_dispatch (MXFP4, FP16 cache, other tiers)
    VERIFY_NVFP4_OVERLAY,   // spec-verify gemm_nvfp4_batched{,_acc} on the GGUF source's NVFP4 overlay
    VERIFY_NVFP4_SOURCE,    // spec-verify gemm_nvfp4_batched{,_acc} on the native NVFP4 source, M <= 4
    SMALLM_A16,             // small-M gemm_nvfp4_smallm, FP16 activations
    SMALLM_A4,              // small-M smallm_launch_a4, NVFP4 activations
    GGUF_IN_PLACE,          // gemm_uncached_in_place on a GGUF source, M <= 32
    DP4A_Q4K,               // gemm_q4k_dp4a_dense, M <= 64
    DP4A_Q5K,               // gemm_q5k_dp4a_dense, M <= 64
    FP16_CACHE,             // gemm() on the FP16 cache entry (cuBLASLt)
    FP8_PREFILL,            // gemm_fp8_prefill_ on the FP8 cache entry
    IMMA_SOURCE,            // imma_source_prefill on a GGUF source with an NVFP4 overlay
    GGUF_SOURCE_DEQUANT,    // gguf_source_fallback on a GGUF source with an NVFP4 overlay
    NVFP4_PROMPT_A16,       // gemm_nvfp4_f16_tile (gemm.nvfp4_prompt_a16_attn)
    NVFP4_TILE,             // CUTLASS_NVFP4 tier own tile GEMM via the kernel registry
    NVFP4_TIER_FP8,         // CUTLASS_NVFP4 tier FP8 cache entry via gemm_cublaslt
    NVFP4_DEQUANT,          // gemm_nvfp4 dequant safety net (native NVFP4 source)
    MXFP4_PREFILL,          // mxfp4_prefill_gemm
    GGUF_SOURCE_NO_SFATOM,  // last route: SfAtom-less CUTLASS_NVFP4 entry, gguf_source_fallback
    GENERIC_DISPATCH,       // last route: gemm_dispatch
    // Fused / sibling exits outside the ladder, recorded per weight at their call site.
    QKV_GEMV_MXFP4,           // gemv_mxfp4_qkv_fused, M=1
    QKV_GEMV_NVFP4,           // gemv_nvfp4_qkv_fused, M=1 (ungated arm and try_attn_qkv_fused_m1_)
    QKV_DP4A,                 // dp4a_qkv_: dispatch_gemv_qkv_fused (+ a dp4a GEMV for a v of another qtype)
    DP4A_MULTI_M1,            // gemv_dp4a_multi, M=1 siblings of mixed qtypes
    DP4A_GEMV,                // dispatch_dp4a_gemv, M=1, one weight
    DP4A_RESIDUAL,            // dispatch_gemv_residual, M=1 with the residual add
    GATE_UP_GEMV_MXFP4,       // gemv_mxfp4_gate_up_fused, M=1
    GATE_UP_GEMV_NVFP4,       // gemv_nvfp4_gate_up_fused, M=1
    GATE_UP_DP4A,             // gemv_gate_up_fused, M=1
    FP16_PAIR_BATCHED,        // gemm_kv_batched / gemm_pair_batched on a fused FP16 copy, M > 1
    SMALLM_MULTI_WS,          // gemm_nvfp4_smallm_ws_multi, sibling launch, M 2..32
    SMALLM_MULTI_V2,          // gemm_nvfp4_smallm_v2_multi_a4, sibling launch, M 2..32
    FP8_SMALLM_MULTI,         // gemm_fp8_smallm_multi, FP8 sibling launch
    FP8_SMALLM,               // gemm_fp8x8_: gemm_fp8_smallm
    FP8_TILE,                 // gemm_fp8x8_: gemm_fp8_tile
    FP8_CUBLASLT,             // gemm_fp8x8_: gemm_cublaslt
    NVFP4_TILE_SWIGLU_QUANT,  // gemm_nvfp4_tile_swiglu_quant: up GEMM + SwiGLU + down quantize epilogue
    NVFP4_PREQUANT,           // gemm_nvfp4_prequant_: down GEMM on the epilogue's NVFP4 activation
    GEMV_MXFP4_RESIDUAL,      // gemv_mxfp4_{,swiglu_,geglu_}residual, M=1
    GEMV_NVFP4_RESIDUAL,      // gemv_nvfp4_{,swiglu_,geglu_}residual, M=1
    GEMV_FP8_RESIDUAL,        // gemv_fp8_residual, M=1
    GDN_INPUT_GEMV_NVFP4,     // gemv_nvfp4_gdn_input_fused, M=1
    GDN_INPUT_GEMV_FP8,       // gemv_fp8_gdn_input, M=1
    F16_NARROW,               // gemm_f16_narrow_{prefill,smallm}: GDN alpha + beta
    COUNT,
};

const char* attn_prefill_path_name(AttnPrefillPath p);
const char* attn_prefill_outer_name(AttnPrefillOuter p);
const char* attn_decode_path_name(AttnDecodePath p);
const char* moe_prefill_outer_name(MoePrefillOuter p);
const char* moe_prefill_path_name(MoePrefillPath p);
const char* dense_path_name(DensePath p);

}  // namespace imp
