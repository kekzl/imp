#pragma once

// One of nine RuntimeConfig sections split from core/dispatch_policy.h:
// isolates a TU that touches only this section from the other eight's churn.
// Pure move, byte-identical; dispatch_policy.h still includes all nine.

#include <cstdint>
#include <string>
#include <vector>

namespace imp::cfg {

struct GDN {
    bool ref_kernel = false;
    // GDN chunkwise SSD scan: dispatches gdn_scan_chunkwise_f32
    // (chunk-cached K/Q in shared memory) instead of the per-token-loop fused
    // scan. Bit-near-equivalent (FP16 1e-3 / FP32 1e-5 tol). Default on;
    // opt out via gdn.chunkwise_scan=false if a model regresses.
    bool chunkwise_scan = true;
    // Chunk-parallel prefill scan: WY per-chunk factors computed on grid
    // (chunks x heads), full-device parallel; only a cheap per-head matmul
    // chain stays sequential. Applies to single-sequence prefill (n>=128),
    // HD=SS=128, either state dtype. Falls back silently if the workspace was not allocated.
    // runtime.deterministic turns it, chunkwise_scan and state_bf16 off: blocks start at the
    // chunk start, BF16 state rounds at chunk ends (#2556; Qwen3.8-27B pp4096 -42 %).
    bool chunkpar_scan = true;
    // Mamba2 prefill scan in the chunked SSD form (fp16 tensor-core GEMMs per 64-token chunk,
    // fp32 accumulate) for chunks >= 128 tokens at state 128; off = ssm_scan_reg.
    // runtime.deterministic turns it off (chunk-partition dependent).
    bool ssd_scan = true;
    // GDN recurrent state as BF16 instead of FP32 (halves state traffic;
    // arithmetic stays FP32 in registers). HD=SS=128 only; forces the fused
    // scan (chunkwise/ref kernels are FP32-state only). FP16 state is refuted
    // (subnormal truncation). Default on; opt out via gdn.state_bf16=false.
    bool state_bf16 = true;
    // Override gated-DeltaNet weight layout.
    std::string layout_override;
};

// Chunks per chunk-parallel strip, 0 = auto (strip in [4,16], fullest last wave); former gdn.chunkpar_strip,
// removed 2026-11-14.
inline constexpr int kGdnChunkparStrip = 0;
}  // namespace imp::cfg
