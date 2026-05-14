#pragma once
// =============================================================================
// mtp_head.h — Multi-Token Predictor head storage
// =============================================================================
//
// Trained MTP head shipped alongside DeepSeek-V3-family models (Qwen3.6,
// DeepSeek V3, etc.) as `model_mtp.safetensors` in the model directory.
//
// Currently scaffolding only (Phase 1.A): the loader detects the file's
// existence and populates `MtpHeadInfo` with size metadata, but the
// actual tensors are not yet uploaded. Wiring forward+verify is documented in
// `docs/superpowers/specs/2026-05-14-mtp-wiring-design.md` (Phases 1.B-5).
//
// Reference architecture (Qwen3.6-NVFP4 MTP):
//   mtp.fc                        4096 → 2048    project concat(emb, h_prev)
//   mtp.pre_fc_norm_embedding     2048           per-input RMSNorm
//   mtp.pre_fc_norm_hidden        2048           per-input RMSNorm
//   mtp.layers.0                  one Qwen3.6 transformer layer (attn + MoE)
//   mtp.norm                      2048           final RMSNorm
//
// LM head is shared with main model (`model.lm_head.weight`).
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <string>

namespace imp {

// Lightweight metadata populated by the loader when an MTP head file is
// detected. Phase 1.A: this is all the loader produces — actual tensors are
// not yet uploaded and the field exists only to signal "this model ships a
// trained MTP head; future work can wire spec-decode against it."
struct MtpHeadInfo {
    std::string path;             // absolute path to model_mtp.safetensors
    size_t      file_bytes = 0;   // on-disk size (informational; for VRAM budget hints)
    int         n_tensors  = 0;   // count parsed from safetensors header (0 if not parsed)
};

// Phase 1.B+ will add: actual Tensor handles for each MTP weight, BF16/FP16/NVFP4
// storage decision, forward kernel hooks. For now the struct stays minimal.

}  // namespace imp
