#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct GEMM {
    // MoE batch prefill via the grouped IMMA kernel (one launch over all
    // experts, gridDim.z=expert, BM=32). Covers Q8_0/Q4_K expert tensors; Q6_K
    // down_proj stays on dequant->cuBLAS. Default on; false re-enables the
    // per-expert fallback arms. Re-measure by 2026-12-01 (AUDIT_ARCH_2026 A1-2).
    bool moe_imma_prefill = true;
    // Extend the NVFP4 decode cache to ALL quantized types (Q4_K, Q3_K, etc.),
    // not just Q8_0/Q6_K/Q5_K. Trades VRAM for decode throughput on sub-8-bit
    // models. Default off.
    bool nvfp4_decode_all = false;
    // Quantize a native-precision LM head to an NVFP4 decode cache ("auto"|"on"
    // |"off", legacy bool accepted). GDN/SSM-hybrid models also get it while
    // nvfp4_lm_head_gdn = true (default, below). auto = ON for native BF16/F16
    // heads and small dense GGUF heads (d_model<=4096); off for larger/MoE GGUF heads.
    // "fp8" = per-row FP8 E4M3 head instead of NVFP4 (F16 or GGUF source, GDN included, #2156).
    std::string nvfp4_lm_head = "auto";
    // FP16-accumulate cuBLAS prefill GEMMs (CUBLAS_COMPUTE_16F): sm_120 runs
    // FP32-accumulate FP16 tensor cores at 1/4 rate, so 16F reaches full rate.
    // "auto" (default) enables per-arch at init: off for GEMMA3/4 (PPL cost)
    // and GPT_OSS (FP16-residual-overflow risk). Applies only to F16xF16->F16, M>1.
    std::string cublas_fp16_acc = "auto";
    // Allow NVFP4 LM head on GDN/SSM-hybrid models too (normally excluded: an
    // older NVFP4 method degraded recurrent-state coherence). Real but small
    // quality cost for a decode gain. Default on; gemm.nvfp4_lm_head=false
    // still kills the NVFP4 LM head entirely (dense + GDN).
    bool nvfp4_lm_head_gdn = true;
    // Batched-decode (n>1) LM head via one NVFP4 tensor-core GEMM (gemm_nvfp4_tile_f32, #2937)
    // instead of the FP16-activation batched-M GEMV: reads the weight once for
    // the batch, but forces NVFP4 activations on the final logits (quality/speed
    // trade). n==1 decode and the spec-verify LM head never take this path. Default on.
    bool nvfp4_lm_head_cutlass = true;
    // Batched-decode LM head at n<=32 on the small-M mxf4nvf4 kernel (FP32
    // logits, final-norm-fused activation quantize) ahead of the own
    // NVFP4 tile GEMM: avoids reading that GEMM's weight sweep below the compute
    // floor at small batch. Same W4A4 numerics as the tile GEMM. Default on.
    bool nvfp4_lm_head_smallm = true;
    // Stream-K on prompt rows (M >= 33) of the own NVFP4 tile: its k split follows M, so a prompt row
    // moves with its chunk (prefix-cache resend). "auto": off on MoE (Gemma-4-26B resend 0.179 nats -> 0,
    // pp512 -0.7 %), on for dense (Qwen3-8B-NVFP4 pp512 -2.1 % without it). Decode rows keep stream-K.
    std::string nvfp4_prompt_streamk = "auto";
    // Own small-M FP8 x FP8 GEMM (src/quant/fp8_gemm_smallm.cu) instead of cuBLASLt for decode rows
    // (n > 1, never a prompt row) up to this M (0 off, max 32). Qwen3-8B-FP8 conc 16 +46 %, conc 4 +61 %.
    int fp8_smallm_max_m = 32;
    // M=1 decode rows on the same kernel (activations quantized with the checkpoint's static input
    // scale, W8A8) instead of the FP16-activation FP8 GEMV. Qwen3-8B-FP8 tg128 +4.8 %.
    bool fp8_smallm_m1 = true;
    // FP8 x FP8 activations take the checkpoint's static input_scale; false (or no input_scale): per-token
    // dynamic scales (row absmax / 448) on the own small-M / tile GEMMs, never a batch-wide absmax.
    bool fp8_static_act_scale = true;
    // Small-M activations packed to NVFP4 (A4) before the GEMM. false keeps them
    // FP16 (gemm_nvfp4_smallm, dequant+HMMA): the M=1 GEMV numerics at 2..32 rows,
    // no sibling-pair launch, no producer-fused quantize. Quality instrument (#2019).
    bool nvfp4_smallm_a4 = true;
    // NVFP4 PREFILL copies of the full-precision GDN projections: M>32 rows run the own NVFP4
    // tile GEMM (W4A4) instead of the FP16 GEMM (~320 TFLOPS on sm_120, 35-40% of peak).
    // "false"/"off" none, "true"/"all" in+gate+out, or a comma list of in|gate|out.
    // M<=32 and M=1 keep FP16 / the FP8 sidecar. Measured: in pp4096 +7.7% (Qwen3.6-35B),
    // +12.3% (Qwen3.8-27B, BF16 in_proj export) for PPL (45k corpus) +0.68% / +0.32..0.56%;
    // gate +0.60%, out +1.98%, all +4.25% PPL on Qwen3.6-35B stay opt-in (roadmap Open 12).
    // Builds only for projections without FP8 decode rows while the MXFP8 route below is fp8rows (default).
    std::string nvfp4_gdn_proj_prefill = "in";
    // MXFP8 (E4M3, UE8M0 per 32) prefill copies of the same projections on the own MXFP8
    // tile GEMM (#2928): 3 mantissa bits per element against NVFP4's 1, twice the copy bytes.
    // Same vocabulary; a projection named here takes MXFP8 ahead of its NVFP4 copy. Qwen3.8-27B
    // 45k PPL: in +0.08%, gate -0.51%, out -0.18% (NVFP4 +0.32 / +0.01 / +0.47%); pp4096
    // Qwen3.6-35B in +6%, all +11.6%. Opt-in: 495 MiB (in) / 990 MiB (all) of copies on
    // Qwen3.6-35B leave its KV pool 50 blocks at stock flags (NVFP4 in, 270 MiB: 225).
    // "fp8rows": no copy, in/gate/out run this GEMM on the FP8 decode sidecar's own rows (SFB = row
    // power of two, residual per column), Phase 4b frees the F16 sources. "auto" = fp8rows
    // (Flash-Next 386 -> 422 slots/layer; Qwen3.6-35B-A3B 26496 -> 23804 MiB, PPL -1.0 %, pp2048 -4.7 %).
    std::string mxfp8_gdn_proj_prefill = "auto";
    // FP8 decode sidecar for full-precision attention projections (wq/wk/wv/wo),
    // same per-row-scale mechanism as the GDN/SSM FP8 decode sidecar, decode-only (M=1).
    // "auto" = on for gpt-oss and Qwen4Exp (dense BF16 weights get no NVFP4 decode
    // cache). Other arches off pending a decode-path PPL gate (--prefill-chunk-size 1).
    std::string fp8_attn_proj = "auto";
};

// Former GEMM keys, values measured once and never moved (keys removed 2026-11-14).
// Own NVFP4 tile GEMM from this M: pp128 Qwen3-8B 42.8-43.3 -> 34.8-35.1 ms vs CUTLASS (#2935).
inline constexpr int kNvfp4TileMinM = 1;
// cuBLASLt block-scaled NVFP4 on the own tile's refusal from this M: M=2048 -5.8..-7.7 % vs CUTLASS (#2540).
inline constexpr int kNvfp4CublasltMinM = 2048;
// Own per-tensor FP8 tile GEMM from this M: Qwen3-8B-FP8 pp4096 +0.6 %, below it cuBLASLt (own pp512 -18 %).
inline constexpr int kFp8TileMinM = 2048;
}  // namespace imp::cfg
