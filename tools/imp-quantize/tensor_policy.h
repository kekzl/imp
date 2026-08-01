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

}  // namespace imp::quantize
