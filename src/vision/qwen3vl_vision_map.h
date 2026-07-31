#pragma once

// Qwen3-VL vision-tower tensor names -> slots.
//
// Split out as a pure function because this is where a vision tower silently
// goes wrong: a misrouted name does not crash, it produces a plausible-looking
// encoder that returns unrelated embeddings. Keeping it free of Tensor and file
// I/O means it can be tested exhaustively against the real name list without a
// checkpoint or a GPU.
//
// Names come from `model.visual.*` in the checkpoint (the `model.visual.`
// prefix is NOT included — callers strip it, as weight_map already does).

#include <string>

namespace imp {

enum class Qwen3VLVisionSlot {
    Unknown = 0,
    PatchEmbedWeight,
    PatchEmbedBias,
    PosEmbed,
    // Per-block (block index in `index`)
    Norm1Weight,
    Norm1Bias,
    QkvWeight,  // fused [3*hidden, hidden]; the loader slices it into q/k/v
    QkvBias,
    ProjWeight,
    ProjBias,
    Norm2Weight,
    Norm2Bias,
    Fc1Weight,
    Fc1Bias,
    Fc2Weight,
    Fc2Bias,
    // Mergers. `index` is -1 for the main merger and 0..n-1 for DeepStack ones.
    MergerNormWeight,
    MergerNormBias,
    MergerFc1Weight,
    MergerFc1Bias,
    MergerFc2Weight,
    MergerFc2Bias,
};

struct Qwen3VLVisionRef {
    Qwen3VLVisionSlot slot = Qwen3VLVisionSlot::Unknown;
    int index = -1;  // block index, or DeepStack merger index; -1 when N/A
};

// Returns slot == Unknown for anything not recognised, so a caller can report
// leftovers instead of silently ignoring them.
Qwen3VLVisionRef qwen3vl_map_vision_tensor(const std::string& name);

}  // namespace imp
