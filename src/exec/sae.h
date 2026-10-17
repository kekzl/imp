#pragma once

// Sparse autoencoder on the residual after one layer (roadmap row 116): JumpReLU SAE in the
// Gemma Scope 2 layout (params.safetensors: w_enc [d_model, width], w_dec [width, d_model],
// b_enc, threshold [width], b_dec [d_model], F32). Encode: a = pre * (pre > threshold),
// pre = x w_enc + b_enc. Clamp feature j to c: x += (c - a_j) w_dec[j].

#include "exec/sae_token.h"
#include "memory/vram_owned.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace imp {

struct SaeHost {
    int layer = -1;  // 0-based decoder layer whose output the SAE reads
    int d_model = 0;
    int width = 0;
    std::vector<float> w_enc, w_dec, b_enc, threshold, b_dec;
};

struct SaeDevice {
    int layer = -1;
    int d_model = 0;
    int width = 0;
    VramOwned<float> w_enc, w_dec, b_enc, threshold, b_dec;
};

// One hidden_ row the SAE runs on during a forward: readout = report its features after the
// step, clamp = (feature, value) edits applied to the residual.
struct SaeRowSpec {
    int row = 0;
    bool readout = false;
    std::vector<std::pair<int32_t, float>> clamp;
};

// Readout per readout row: dense activations (after clamps) and the reconstruction error of the
// unclamped residual, ||x - decode(encode(x))|| / ||x||.
struct SaeReadout {
    int rows = 0;
    int width = 0;
    std::vector<float> acts;  // [rows, width]
    std::vector<float> rel_err;
};

// Reads dir/config.json (hf_hook_point_in "model.layers.N.output", width) and
// dir/params.safetensors. false + *err on a missing file, a layer outside [0, n_layers), a
// width or d_model mismatch, or a non-F32 tensor.
[[nodiscard]] bool load_sae_dir(const std::string& dir, int d_model, int n_layers, SaeHost& out,
                                std::string* err);

}  // namespace imp
