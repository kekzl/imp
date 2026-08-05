#pragma once

// The two refusals the mmproj GGUF path was missing.
//
// This loader speaks exactly one dialect: the LLaVA/SigLIP layout that Gemma-3
// (`gemma3`) and Gemma-4v (`gemma4v`) export — separate attn_q/attn_k/attn_v,
// one projection tensor, no side towers. A checkpoint in another dialect does
// not fail to parse. It parses *partly*: the names it shares land in their
// slots, the names it does not are logged at DEBUG and dropped, and the model
// comes back looking loaded. The encoder then hands the null slots straight to
// vision_gemm.
//
// Qwen3-VL's mmproj is the live example — 247 of its 316 tensors land, and the
// 69 that do not are precisely what makes it Qwen3-VL: fused `attn_qkv` (48),
// the DeepStack mergers (18), the second projector layer `mm.2` (2) and the
// temporal half of the patch conv (1). imp reads that model from SafeTensors
// instead; nothing here can.
//
// So there are two gates, and the second is the one that matters longer-term:
// name the dialect we know we cannot read, and behind it, verify that every
// slot the encoder dereferences unconditionally actually got filled — which
// catches the dialects nobody has named yet.

#include "vision/vision_model.h"

#include <string>

namespace imp {

// Empty when this loader can read `projector_type`; otherwise the reason,
// phrased for whoever has to pick a different flag.
std::string vision_projector_reject_reason(const std::string& projector);

// Empty when every slot the encoder dereferences unconditionally is filled;
// otherwise the GGUF name of the first one that is not.
//
// The predicate is `ndim == 0`, not `data == nullptr`: the probe walks the same
// loader with uploads counted rather than performed, so in that pass every slot
// has a null pointer and a real shape. Shape is what "was assigned" means here.
//
// Optional slots are deliberately absent from the walk — the encoder guards
// patch_embd_b, the attention and FFN biases, post_ln and mm_post_norm with an
// `if (.data)`, and gemma4v has neither LN biases nor attention biases at all.
std::string vision_model_missing_slot(const VisionModel& model);

}  // namespace imp
