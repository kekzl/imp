#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct GDN {
    bool fp32_scan = false;
    bool fp32_out = false;
    float norm_eps_override = 0.0f;  // 0 = use model default
    bool ref_kernel = false;
    bool vhead_reorder = false;
    // GDN chunkwise SSD scan: dispatches gdn_scan_chunkwise_{f32,fp32out}
    // (chunk-cached K/Q in shared memory) instead of the per-token-loop fused
    // scan. Bit-near-equivalent (FP16 1e-3 / FP32 1e-5 tol). Default on;
    // opt out via gdn.chunkwise_scan=false if a model regresses.
    bool chunkwise_scan = true;
    // Chunk-parallel prefill scan: WY per-chunk factors computed on grid
    // (chunks x heads), full-device parallel; only a cheap per-head matmul
    // chain stays sequential. Applies to single-sequence prefill (n>=128),
    // HD=SS=128, either state dtype. Falls back silently if the workspace was not allocated.
    bool chunkpar_scan = true;
    // Chunks (64 tokens each) per chunk-parallel strip: kernel 1 launches
    // strip*n_heads CTAs at one CTA/SM, so strip sets the wave quantization.
    // 0 = auto (strip in [4,16], fullest last wave); 1..16 pins it.
    int chunkpar_strip = 0;
    // GDN recurrent state as BF16 instead of FP32 (halves state traffic;
    // arithmetic stays FP32 in registers). HD=SS=128 only; forces the fused
    // scan (chunkwise/ref kernels are FP32-state only). FP16 state is refuted
    // (subnormal truncation). Default on; opt out via gdn.state_bf16=false.
    bool state_bf16 = true;
    // Batched decode (2<=M<=32): alpha/beta projections run as one split-K
    // FP16 tensor-core launch instead of two cuBLAS GEMMs. FP16-resident
    // alpha/beta weights only; other tiers keep the two-call path.
    bool alpha_beta_smallm = true;
    // Single-stream decode (M=1) on native-NVFP4 hybrids: in_proj/gate/alpha/
    // beta fused into one GEMV, out-proj adds the residual in its epilogue,
    // gated attention runs q|k|v as one launch. false = per-projection launches.
    bool m1_fused = true;
    // Override gated-DeltaNet weight layout.
    std::string layout_override;
};
}  // namespace imp::cfg
