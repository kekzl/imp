#pragma once

// GEMM configuration, one of the nine sections split out of
// core/dispatch_policy.h on 2026-08-21.
//
// WHY. dispatch_policy.h aggregates all nine and is included by 23 translation
// units, of which 21 touch two sections or fewer. Adding one field to it costs
// 137.1 s of incremental rebuild, against 9.1 s for a small .cpp and 14.6 s for
// the largest .cu the file-size gate polices. A TU that needs only this section
// can include only this header and stop rebuilding when the others change.
//
// This is F-10 one level down, and dispatch_policy.h's own preamble records the
// original: config.h was included by 22 files, 85 TUs transitively, and changed
// 130 times in six months - "the highest build cost in the repo". Lifting nine
// sections into an aggregate fixed that, and gave the aggregate the same
// property for the same reason.
//
// Pure move: the contents below are byte-identical to their previous form, and
// dispatch_policy.h includes every one of these, so no existing include breaks.

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
    // Q8_0 INT8 IMMA prefill GEMM (mmq_q8_imma.cu): fused dequant on the
    // int8 tensor cores (s8.s8.s32 measured 968 TOPS — full rate, unlike
    // the quartered f32-accumulate paths). Replaces the dequant-to-FP16 →
    // cuBLAS round-trip for Q8_0 prefill (M ≥ 64). Redesigned against the
    // Q4_K-IMMA phase-2B ceiling diagnosis (SMEM-staged scales, 128x128x64
    // tiles, symmetric epilogue). Default on.
    bool q8_imma_enabled = true;
    // Q4_K dense prefill via the (new-stack) IMMA kernel: uses
    // mmq_q4k_imma_reorder's symmetric-s8 + α/β form with the unified
    // β·rowsum epilogue. Experimental: default off. (Distinct from the
    // retired 2026-05 64x32 q4k_imma kernel that plateaued at 40 TOPS.)
    bool q4k_imma_prefill = false;
    // MoE batch prefill via the grouped IMMA kernel (one launch over all
    // experts, gridDim.z = expert, BM=32 small-M tile for the typical
    // ~32-rows-per-expert routing at pp512). Covers Q8_0/Q4_K expert
    // tensors; others (Q6_K down_proj) stay on dequant→cuBLAS. This is
    // lever #1 for the 2.4-2.6x GGUF-MoE prefill gap
    // (docs/archive/prefill_gap_2026_06_07.md §4.2). Default on.
    bool moe_imma_prefill = true;
    // Extend NVFP4 decode cache to ALL quantized types (Q4_K, Q3_K, etc.),
    // not just the default Q8_0/Q6_K/Q5_K set. Trades VRAM for decode
    // throughput on sub-8-bit models (e.g. Gemma-3-12B Q4_K_M: dp4a GEMV
    // at 130 tok/s → NVFP4 kpar GEMV target ~165 tok/s).
    bool nvfp4_decode_all = false;
    // Quantize a native-precision (FP16/BF16) LM head to an NVFP4 decode
    // cache. Native-NVFP4 checkpoints (llm-compressor/Modelopt) store
    // lm_head in BF16, so decode pays a cuBLAS FP16 GEMV over the
    // vocab×d_model matrix (~0.78 ms/token, ~19% of decode on Qwen3-8B).
    // The GGUF path already NVFP4-caches a Q*_K/Q8_0 output_proj; this
    // extends the same win to native-NVFP4 dense models. Excluded for
    // GDN/SSM-hybrid models (LM-head NVFP4 degrades recurrent-state
    // quality — see memory lm_head_only_nvfp4_qwen3_6_refuted).
    // "auto" (#982 net rule) | "on" | "off" (legacy true/false accepted).
    // auto = ON for native BF16/F16 heads (4x byte win, +8-16% decode,
    // +2.2% PPL, GOAL-listed trade) and for small dense GGUF heads
    // (d_model <= 4096: 4B/8B measured net-positive); OFF for larger or
    // MoE GGUF heads where the 2026-07-12 parity sweep measured the PPL
    // cost above the decode win (14B +1.9%/+2.1%, 30B-A3B +3.7%/+5.0%).
    std::string nvfp4_lm_head = "auto";
    // FP16-accumulate cuBLAS prefill GEMMs (CUBLAS_COMPUTE_16F instead of
    // 32F). GeForce sm_120 runs FP16 tensor cores with FP32 accumulate at
    // 1/4 rate (measured 2026-06-07: 253 vs 1956 TFLOPS saturated
    // mma.sync); the cuBLAS 32F prefill GEMMs sit at ~225 TFLOPS — ~89% of
    // that quarter-rate ceiling, so the kernel is fine, the compute type
    // is the cap. 16F measured +24.9% q8 pp512 model-level (2026-06-07,
    // paired same-day restarts), decode neutral. "auto" (default) enables
    // it per-arch at engine init: ON except GEMMA3/GEMMA4 (measured +0.7%
    // PPL on gemma-3-12b) and GPT_OSS (known FP16-residual-overflow
    // sensitivity — f16 accumulators are the same hazard class). "on"
    // forces it everywhere, "off" restores 32F accumulate. Legacy bool
    // values (true/false/1/0) parse as on/off. Applies only to
    // F16xF16→F16 with M>1 (prefill); decode GEMV and mixed-precision
    // paths are untouched. Standalone tools that skip engine init treat
    // "auto" as off.
    std::string cublas_fp16_acc = "auto";
    // Allow NVFP4 LM head even on GDN/SSM-hybrid models (normally excluded —
    // an older NVFP4 method degraded recurrent-state coherence, memory
    // lm_head_only_nvfp4_qwen3_6_refuted). Quantified 2026-05-29 with the
    // current quantize-FP16→NVFP4 path + the new `imp-cli --perplexity` tool
    // on Qwen3.6-35B: decode +11.4% (219.6→244.7 tok/s; the 248k-vocab
    // lm_head is ~14% of decode) at a small but REAL quality cost —
    // perplexity 15.90 (FP16) → 16.25 (NVFP4), +2.2% (FP16 PPL is stable
    // run-to-run, so it's signal not noise). Default ON: the +11.4% decode
    // gain serves the primary mission metric (best batch=1 tok/s on the
    // 5090) and the +2.2% PPL cost is small; set false to keep the FP16
    // lm_head for maximum coherence. gemm.nvfp4_lm_head=false still kills
    // the NVFP4 lm_head entirely (dense + GDN).
    bool nvfp4_lm_head_gdn = true;
    // Batched-decode (n>1) LM head via a single CUTLASS NVFP4 tensor-core
    // GEMM instead of the FP16-activation batched-M GEMV. Reads the LM-head
    // weight once for the whole batch (vs ceil(n/4)x for the GEMV), but the
    // FP4×FP4 MMA forces NVFP4 activations on the final logits (the GEMV kept
    // FP16 activations) — a quality/speed trade. Costs ~vocab*d_model/16 B of
    // SfAtom scales (FP4 data borrowed from the decode cache; only allocated
    // when max_batch_size > 1). Default ON since the 2026-07-12 PPL sweep
    // (teacher-forced, 13.5k-token corpus, imp-cli --perplexity): MoE/hybrid
    // +1.9-2.1% PPL (Coder-30B 10.19→10.38, Modelopt-30B 11.65→11.88,
    // Qwen3.6-35B 13.39→13.66), dense +0.2-0.5% (within the ±0.3-0.5%
    // run-to-run spread), for +8% aggregate concurrent decode @16 — the
    // 1173-tok/s Coder-30B headline. Single-stream (n==1) output is
    // bit-identical either way: the n==1 decode GEMV and the spec-verify
    // LM head (for_each_lm_head_batch_ allow_cutlass=false) never take
    // this path. Set false for maximum batched-serving coherence.
    bool nvfp4_lm_head_cutlass = true;
    // Small-M (<=32) NVFP4 GEMM for batched decode (impl selected below).
    // History: the W4A16 dequant+HMMA v1 won isolated (23.9 vs CUTLASS's
    // 41.4 us in-situ on N=5120) and LOST the real 32-stream step (45.8 us,
    // -11% aggregate) — its synchronous SIMT loads are exposed to the GDN
    // scan's L2 pressure, and the A4 variant proved footprint was not the
    // driver (742-747 vs 955-963 tok/s). The v2 kernel closes exactly that
    // hole: native block-scaled mxf4nvf4 MMA fed by a producer/consumer
    // cp.async+mbarrier pipeline on the SAME plain weight bytes the M=1
    // GEMVs read — zero extra VRAM (unlike the discarded Marlin W4A16
    // sidecar, PR #1764, which needed a repacked second weight copy and
    // capped out at 13% coverage on the 27B). Measured 2026-08-25,
    // Qwen3.8-27B-NVFP4, mbs=32/seq4096, alternating A/B, 3 trials each:
    // 32 streams 992.5 -> 1151.7 tok/s aggregate (+16.0%, all 9 ON waves
    // above all 9 OFF waves), 8 streams 363.8 -> 494.6 (+36.0%); isolated
    // M=32 N=5120 K=5120: 10.4 us vs CUTLASS 41.4 in-situ (weight floor
    // 8.2). degen_suite 50/0 ON. Default ON since the v2 A/B.
    bool nvfp4_smallm = true;
    // Which small-M implementation the gate above dispatches: 1 = the W4A16
    // dequant+HMMA kernel (kept for A/B), 2 = the native mxf4nvf4
    // producer/consumer pipeline (nvfp4_gemm_smallm_v2.cu; isolated 10.4 us
    // on M=32 N=5120 K=5120 vs v1's 23.9 and CUTLASS's 41.4 in-situ).
    int nvfp4_smallm_impl = 2;
    // (gemm.nvfp4_ssm_proj — the 2026-05-30 opt-in that forced GGUF-hybrid
    // GDN projections into the NVFP4 decode cache — was REMOVED 2026-07-11:
    // it had bit-rotted in the tier refactors (measured 71 tok/s vs its
    // original 248 on Qwen3.6-35B Q4_K_M) and is superseded by the GGUF
    // branch of fp8_ssm_proj below, which is faster than the flag ever was
    // and quality-safer than 4-bit into the recurrent scan. The recurrent
    // in_proj/out_proj stay OUT of the NVFP4 decode cache unconditionally:
    // they feed the GDN/SSM scan, which accumulates quantization error in
    // the state H across tokens.)
    // Native-NVFP4 hybrid models store SOME projections BF16 because the
    // Modelopt/llm-compressor recipe excluded them from NVFP4. At decode these
    // run as FP16 GEMVs (gemv_fp16_kernel). This opt-in quantizes the
    // recipe-excluded BF16 **attention q/k/v/o** to an NVFP4 decode-cache
    // entry at init (direct quantize_fp16_to_nvfp4_async, mirroring
    // nvfp4_lm_head_gdn). q/k/v/o are stateless within a step → low quality
    // risk. MEASURED (2026-05-30): Nemotron-3-Nano-30B **+3.8% decode**,
    // perplexity-neutral (within noise). No-op on models whose attention is
    // already NVFP4 (e.g. Qwen3.6-35B). Default false.
    //
    // NOTE: the analogous lever for the BF16 GDN/Mamba in_proj/out_proj was
    // built and MEASURED to REGRESS decode −9% (Nemotron) to −20% (Qwen3.6) —
    // the tuned FP16 GEMV (70-81% HBM) beats the NVFP4 GEMV for the wide
    // GDN-output shapes, so the bandwidth saving never materializes. Keeping
    // those projections FP16 is correct for SPEED (not just quality); no flag
    // is provided for them. (See docs/MTP/SafeTensors profiling notes.)
    bool nvfp4_attn_proj = false;
    // FP8 (E4M3, per-tensor amax scale) DECODE sidecar for the native-
    // precision GDN/Mamba in_proj/out_proj that the NVFP4 lever above
    // regresses on: FP8 halves the FP16 GEMV bytes with byte-aligned dense
    // loads (none of the 4-bit packing overhead that made NVFP4 lose to the
    // tuned FP16 GEMV on the wide GDN shapes), and its quality risk into
    // the recurrent scan is far smaller than 4-bit. Prefill and the M>1
    // verify chunks keep the resident FP16 source (quality); only the M=1
    // decode GEMV dispatches the FP8 copy (gemv_fp8_rowscale, per-row
    // scales — one per-tensor scale over the heterogeneous GDN input pack
    // cost +4% PPL; per-row is PPL-flat). Costs +0.5 byte/elem VRAM for
    // the sidecar copies.
    // GGUF hybrids: also covers Q8_0-source ssm_in/gdn_gate/ssm_out
    // (UD-Q4_K_M keeps exactly these at Q8_0). Those handles were in no
    // decode cache at all (phase-3 quality lock) → every decode token
    // paid a full dequant→cuBLAS round-trip; the FP8 copy is byte-neutral
    // vs Q8_0 but runs the tuned rowscale GEMV. Sub-8-bit sources are
    // excluded (FP8 would increase decode bytes and stack rounding on a
    // coarse lattice). No-op on 27B-class checkpoints whose SSM
    // projections are already native NVFP4.
    // MEASURED (2026-07-10): Qwen3.6-35B-A3B-NVFP4 decode +19% (268.6→320.3
    // tok/s spec-off, 261→308 with default spec), PPL flat (8.021→8.012);
    // Nemotron-3-Nano PPL flat (4.184→4.117).
    // MEASURED (2026-07-11, GGUF branch): Qwen3.6-35B UD-Q4_K_M decode +21%
    // (224.4→272.0 defaults, 219.2→265.9 spec-off), PPL 4.215→4.289 (+1.8%
    // on the 201-token corpus — E4M3 stacked on the Q8_0 lattice; an
    // accepted trade like nvfp4_lm_head_gdn, set false to revert),
    // degen_suite 33/33. Default ON.
    bool fp8_ssm_proj = true;
    // FP8 decode sidecar for FULL-PRECISION attention projections
    // (wq/wk/wv/wo), same per-row-scale mechanism as fp8_ssm_proj:
    // decode-only (M=1 GEMV), prefill keeps the full-precision source.
    // "auto" = ON only for gpt-oss (#984): its BF16 SafeTensors dense
    // weights get no NVFP4 decode cache (nvfp4_beneficial is GGUF-only),
    // so q/k/v/o decode as 2 B/elem FP16 GEMVs = 33.5% of the decode
    // window at ~1.1 TB/s — halving the bytes is ~halving that time.
    // Other arches stay off pending a PPL gate (the phase-2 comment's
    // attn-weight precision concern was measured on PREFILL compounding;
    // decode-only per-row FP8 is the #949 recipe).
    std::string fp8_attn_proj = "auto";
    // Route native-NVFP4 (Modelopt/llm-compressor) MoE expert DECODE (M=1)
    // through the fast per-expert gemv_nvfp4_moe kernels by borrowing the
    // already-resident contiguous expert data + scales, instead of the
    // CUTLASS grouped-GEMM (which under-utilizes the GPU at M=1). +54-80%
    // MoE decode on Qwen3-30B-A3B / Coder-30B / Gemma-4-26B. Prefill stays
    // on CUTLASS.
    bool nvfp4_moe_decode = true;
};
}  // namespace imp::cfg
