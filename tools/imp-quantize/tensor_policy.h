#pragma once

// Which tensors imp-quantize may touch, and which checkpoints it must refuse. Lives apart
// from main.cpp because it's already been wrong in the dangerous direction: #1159's
// "MoE unquantized" check covered only 3-D stacks, so the HF-standard per-expert 2-D layout
// quantized into a checkpoint that loaded and emitted garbage. Fix was learning WHICH roles
// stay full precision, not a shape check - a rule with that history belongs where a test can reach it.

#include "model/safetensors_raw.h"

#include <string>
#include <vector>

namespace imp::quantize {

// True when this tensor is a 2-D linear matrix the runtime reads through the
// NVFP4 GEMM path. On false, `why_not` says why — the caller prints it and
// copies the tensor through untouched.
bool should_quantize(const RawTensor& t, bool quantize_lm_head, std::string& why_not);

// What happens to an FP8 weight paired with a block-scale grid: the pair is one unit, the
// grid is CONSUMED either way (a grid outliving its weight says nothing true, and raw E4M3
// bytes without their scales are still valid E4M3 meaning something else, undetectably). A
// refused weight is widened to full precision instead and the caller declares it unquantized.
// Exists because the pairing pass and the writer's should_quantize call could disagree (a
// DeepSeek-V3 checkpoint has every MLA latent projection hit this). `gated` is the caller's
// knowledge that a q_proj carries a fused output gate; should_quantize sees one tensor and
// can't know that.
enum class Fp8SourceAction { Quantize, WidenToFullPrecision };
Fp8SourceAction fp8_source_action(const RawTensor& weight, bool gated, bool quantize_lm_head,
                                  std::string& why_not);

// Bytes an [N,K] matrix occupies once quantized: packed nibbles [N,K/2], FP8 micro-scales
// [N,K/16], one F32 tensor scale. Exists so --dry-run and the real write agree by construction:
// the writing path checks its actual buffers against this and fails loudly on a mismatch.
size_t nvfp4_output_bytes(int64_t N, int64_t K);

// MoE experts stored as one 3-D [n_experts,N,K] stack rather than a 2-D tensor per expert
// (gpt-oss mlp.experts.gate_up_proj_blocks, Gemma-4 experts.gate_up_proj): found by inspection
// since they aren't named `.weight`, so the rank check in should_quantize rejects them first
// with no counter or message, and they'd be copied through as BF16 while hf_quant_config.json
// claimed a quantized checkpoint. Returns pointers so the caller can report which tensors and
// what share of the checkpoint they are.
std::vector<const RawTensor*> find_stacked_expert_tensors(const std::vector<RawTensor>& tensors);

// Attention q_proj weights carrying a fused output GATE alongside Q (Qwen3.5/Qwen3-Next
// attn_output_gate): the Q half feeds attention scores, the gate half feeds a sigmoid
// multiplying the output. NVFP4 in the gate half is the measured root cause of #1273 (hybrid
// checkpoints degrading 2.08x-6x, dense 1.05x): rounding only that half reproduces the real
// defect (+0.0169 vs +0.0156 divergence per attention block) while rounding Q sits below noise.
// E2M1 offers 8 magnitudes per micro-block, coarsest near zero, exactly where a sigmoid is most
// sensitive. Detected from shapes (a gated q_proj emits twice what o_proj consumes), not a
// config flag, so it holds for any layout. Callers use this to WARN only: excluding just the
// gate half needs partial quantization (doesn't exist yet).
std::vector<const RawTensor*> find_fused_gate_q_projections(const std::vector<RawTensor>& tensors);

}  // namespace imp::quantize
