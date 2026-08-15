#pragma once

// Which tensors imp-quantize may touch, and which checkpoints it must refuse.
//
// This lives apart from main.cpp because it is the part that has already been
// wrong in the dangerous direction. #1159: "MoE is left unquantized" described
// 3-D stacks only, so the HF-standard per-expert 2-D layout was quantized into
// a checkpoint that loaded and then emitted garbage. The fix was not a shape
// check — it was learning WHICH roles must stay full precision. A rule with
// that history belongs somewhere a test can reach it.

#include "model/safetensors_raw.h"

#include <string>
#include <vector>

namespace imp::quantize {

// True when this tensor is a 2-D linear matrix the runtime reads through the
// NVFP4 GEMM path. On false, `why_not` says why — the caller prints it and
// copies the tensor through untouched.
bool should_quantize(const RawTensor& t, bool quantize_lm_head, std::string& why_not);

// Bytes an [N, K] matrix occupies once quantized: the packed nibbles [N, K/2],
// the FP8 micro-scales [N, K/16], and one F32 tensor scale.
//
// Exists so the --dry-run forecast and the real write agree by construction
// rather than by two people keeping two expressions in step. The writing path
// checks its actual buffers against this and fails loudly on a mismatch, so a
// layout change cannot silently turn the forecast into a guess.
size_t nvfp4_output_bytes(int64_t N, int64_t K);

// MoE experts stored as one 3-D [n_experts, N, K] stack rather than a 2-D
// tensor per expert.
//
// These have to be found by inspection rather than by the rank check in
// `should_quantize`, because they are not named `.weight` — gpt-oss ships
// `mlp.experts.gate_up_proj_blocks`, Gemma-4 `experts.gate_up_proj` — and the
// name gate rejects them first, as "not a .weight tensor", with no counter and
// no message. They were then copied through as BF16 while `hf_quant_config.json`
// announced a quantized checkpoint: the experts are most of the bytes, so the
// output was a model that had barely shrunk and claimed otherwise.
//
// Returns pointers into `tensors` so the caller can report both which they are
// and how much of the checkpoint they represent — the share is the difference
// between "the experts are the whole model" and "this is an MTP sidecar", and
// the refusal should not guess which.
std::vector<const RawTensor*> find_stacked_expert_tensors(const std::vector<RawTensor>& tensors);

// Attention `q_proj` weights that carry a fused output GATE alongside Q
// (Qwen3.5 / Qwen3-Next `attn_output_gate`). Such a tensor holds two roles: the
// Q half feeds attention scores, and the gate half feeds a sigmoid that
// multiplies the attention output.
//
// It matters here because NVFP4 in the gate half is the measured root cause of
// #1273 — hybrid checkpoints degrading 2.08x-6x while dense ones cost 1.05x.
// Rounding ONLY that half through NVFP4 on a healthy GGUF twin reproduces the
// real defect (+0.0169 divergence injected per attention block against the
// +0.0156 the actual NVFP4 checkpoint injects), while rounding the Q half sits
// below the noise floor. It is not "4-bit is bad": the same half in Q4_K is
// healthy. E2M1 offers 8 magnitudes per micro-block and is coarsest near zero,
// which is exactly where a sigmoid is most sensitive.
//
// Detected from shapes rather than from a config flag so the rule holds for any
// checkpoint layout: a gated `q_proj` emits twice what the layer's `o_proj`
// consumes. Callers use this to warn — excluding the gate half needs partial
// quantization, which does not exist yet, and excluding the whole tensor is a
// compression trade the caller should make deliberately.
std::vector<const RawTensor*> find_fused_gate_q_projections(const std::vector<RawTensor>& tensors);

}  // namespace imp::quantize
