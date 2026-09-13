#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct GEMM {
    bool no_dp4a_gemv = false;
    bool no_dp4a_lm = false;
    bool no_mmvq = false;
    bool no_mmvq_q8_0 = false;
    // Q4_K x FP16 HMMA GEMM: in-SMEM nibble decode + FP16 tensor core
    // m16n8k16 tile kernel. Phase 0 scaffold (default off). When enabled,
    // prefill (M >= 32) Q4_K weights bypass dequant-to-FP16 + cuBLAS.
    bool q4k_hmma_enabled = false;
    // Q8_0 INT8 IMMA prefill GEMM: fused dequant on int8 tensor cores
    // (s8.s8.s32, full rate unlike the quartered f32-accumulate paths),
    // replacing dequant-to-FP16 -> cuBLAS for Q8_0 prefill (M>=64). Default on.
    bool q8_imma_enabled = true;
    // Q4_K dense prefill via the IMMA kernel (symmetric-s8 + alpha/beta form,
    // unified beta*rowsum epilogue). Experimental, default off. Distinct from
    // the retired 64x32 q4k_imma kernel (plateaued at 40 TOPS).
    bool q4k_imma_prefill = false;
    // MoE batch prefill via the grouped IMMA kernel (one launch over all
    // experts, gridDim.z=expert, BM=32). Covers Q8_0/Q4_K expert tensors; Q6_K
    // down_proj stays on dequant->cuBLAS. Default on; false re-enables the
    // per-expert fallback arms. Re-measure by 2026-12-01 (AUDIT_arch_2026 A1-2).
    bool moe_imma_prefill = true;
    // Extend the NVFP4 decode cache to ALL quantized types (Q4_K, Q3_K, etc.),
    // not just Q8_0/Q6_K/Q5_K. Trades VRAM for decode throughput on sub-8-bit
    // models. Default off.
    bool nvfp4_decode_all = false;
    // Quantize a native-precision LM head to an NVFP4 decode cache ("auto"|"on"
    // |"off", legacy bool accepted). Excluded for GDN/SSM-hybrid models (NVFP4
    // LM head degrades recurrent-state quality). auto = ON for native BF16/F16
    // heads and small dense GGUF heads (d_model<=4096); off for larger/MoE GGUF heads.
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
    // Batched-decode (n>1) LM head via one CUTLASS NVFP4 tensor-core GEMM
    // instead of the FP16-activation batched-M GEMV: reads the weight once for
    // the batch, but forces NVFP4 activations on the final logits (quality/speed
    // trade). n==1 decode and the spec-verify LM head never take this path. Default on.
    bool nvfp4_lm_head_cutlass = true;
    // Batched-decode LM head at n<=32 on the small-M mxf4nvf4 kernel (FP32
    // logits, final-norm-fused activation quantize) ahead of the CUTLASS
    // 128-row tile: avoids reading the CUTLASS weight sweep below the compute
    // floor at small batch. Same W4A4 numerics as CUTLASS. Default on.
    bool nvfp4_lm_head_smallm = true;
    // Small-M (<=32) NVFP4 GEMM for batched decode. v1 (W4A16 dequant+HMMA)
    // won isolated but lost the real 32-stream step: synchronous SIMT loads
    // are exposed to the GDN scan's L2 pressure. v2 uses native block-scaled
    // mxf4nvf4 MMA via a producer/consumer cp.async+mbarrier pipeline on the
    // same plain weight bytes the M=1 GEMVs read (zero extra VRAM). Default on.
    bool nvfp4_smallm = true;
    // Stream-K tile scheduler for the CUTLASS NVFP4 prefill GEMM (cooperative
    // 128x128 tile, N>2048): 0=off, 1=heuristic on grids with a short tail wave
    // (>=1 wave, last wave <= half full), 2=forced at every shape. Output bit-identical.
    int nvfp4_cutlass_streamk = 1;
    // Which small-M implementation the gate above dispatches: 1 = W4A16
    // dequant+HMMA kernel (kept for A/B), 2 = native mxf4nvf4 producer/consumer
    // pipeline (nvfp4_gemm_smallm_v2.cu).
    int nvfp4_smallm_impl = 2;
    // Sibling-pair launch for the v2 kernel: two weights consuming the SAME
    // activation (FFN gate|up, GDN in|z) run as ONE launch, same CTA body,
    // bit-identical per tensor. Saves one launch's overhead plus a tail wave per pair.
    bool nvfp4_smallm_pair = true;
    // Batched-decode residual accumulation on the CUTLASS_NVFP4 tier: o/down/
    // GDN-out at 2..32 rows with beta=1 straight into hidden, vs GEMM-to-scratch
    // + elementwise add. Default off: measured negative, since the scratch+add
    // already overlaps the next layer's independent GEMMs (wall-free), while the
    // accumulate path moves those bytes into the critical GEMM's epilogue.
    bool nvfp4_residual_beta1 = false;
    // gemm.nvfp4_ssm_proj was REMOVED (superseded by fp8_ssm_proj's GGUF
    // branch). Recurrent in_proj/out_proj stay OUT of the NVFP4 decode cache
    // unconditionally: quant error accumulates in the scan state H.
    // This flag NVFP4-quantizes recipe-excluded BF16 attention q/k/v/o instead
    // (stateless per step, low risk). The analogous GDN in_proj/out_proj lever
    // regresses decode (tuned FP16 GEMV wins on wide GDN shapes); no flag exists for it. Default false.
    bool nvfp4_attn_proj = false;
    // FP8 (E4M3, per-tensor amax) decode sidecar for native-precision GDN/Mamba
    // in_proj/out_proj (where NVFP4 above regresses): halves FP16 GEMV bytes
    // with dense byte-aligned loads, lower quality risk than 4-bit into the
    // recurrent scan. M=1 decode only (per-row scales); prefill/M>1 keep FP16.
    // GGUF hybrids: also covers Q8_0-source ssm_in/gdn_gate/ssm_out. Sub-8-bit
    // sources excluded. Default on.
    bool fp8_ssm_proj = true;
    // FP8 decode sidecar for full-precision attention projections (wq/wk/wv/wo),
    // same per-row-scale mechanism as fp8_ssm_proj, decode-only (M=1).
    // "auto" = on only for gpt-oss (dense BF16 weights get no NVFP4 decode
    // cache, so halving FP16 GEMV bytes matters most there). Other arches off pending a PPL gate.
    std::string fp8_attn_proj = "auto";
    // Route native-NVFP4 MoE expert decode (M=1) through the fast per-expert
    // gemv_nvfp4_moe kernels (borrowing resident expert data+scales) instead of
    // the CUTLASS grouped-GEMM, which under-utilizes the GPU at M=1. Prefill stays on CUTLASS.
    bool nvfp4_moe_decode = true;
};
}  // namespace imp::cfg
