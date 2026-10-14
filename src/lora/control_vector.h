#pragma once

// Control vectors (roadmap row 114): h += scale * v_l after layer l, no finetune.
// File format = llama.cpp `cvector-generator` GGUF: tensors `direction.N` (F32 [d_model]),
// N >= 1, added after 0-based layer N; layer 0 carries none.

#include "memory/vram_owned.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace imp {

struct ControlVectorHost {
    int d_model = 0;
    // [n_layers][d_model] row-major; a layer without a direction is all zero.
    std::vector<float> dirs;
    // layer_mask[l] = file carries direction.l
    std::vector<uint8_t> layer_mask;
};

// Device copy owned by the engine: dirs [n_layers, d_model] F32, layer_mask as on the host.
struct ControlVectorDevice {
    VramOwned<float> dirs;
    std::vector<uint8_t> layer_mask;
    int d_model = 0;
};

// Parses a control-vector GGUF held in memory. false + *err on a malformed file, a non-F32
// tensor, a width != d_model, or a layer index outside [1, n_layers).
[[nodiscard]] bool parse_control_vector_gguf(std::span<const uint8_t> bytes, int d_model, int n_layers,
                                             ControlVectorHost& out, std::string* err);

// Reads `path` and parses it (same contract).
[[nodiscard]] bool load_control_vector_file(const std::string& path, int d_model, int n_layers,
                                            ControlVectorHost& out, std::string* err);

}  // namespace imp
