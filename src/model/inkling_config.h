#pragma once
// Inkling2-Loop text model shape (config.json, read by model/inkling_weights.cu). CUDA-free: the engine's
// KV sizing and the CPU tests read it too.

#include <vector>

namespace imp {

struct InklingConfig {
    int n_layers = 24, hidden = 2304, n_heads = 24, n_kv_heads = 8, head_dim = 128;
    int d_rel = 16, rel_extent = 1024, window = 1024, sconv_kernel = 4;
    int vocab = 200058, vocab_padded = 201024;
    int n_experts = 128, top_k = 8, inter = 1152;
    int loop_start = 6, loop_end = 18, passes = 3, gate_rank = 64, block_layers = 6;
    float eps = 1e-6f, qk_clip = 22.627417f, rel_clip = 8.0f;
    float log_alpha = 0.1f, log_floor = 8192.0f;
    float gate_cap = 4.0f, up_cap = 25.0f, route_scale = 8.0f, logit_cap = 30.0f;
    std::vector<bool> global;  // full attention layers; the rest attend the last `window` tokens

    // Layer applications per token, and the KV / conv cache index of (layer, pass).
    int n_apps() const { return n_layers + (passes - 1) * (loop_end - loop_start); }
    int app_index(int layer, int pass) const {
        return pass == 0 ? layer : n_layers + (pass - 1) * (loop_end - loop_start) + (layer - loop_start);
    }
    // Weight layer of application a (the inverse of app_index over the passes).
    int app_layer(int a) const {
        return a < n_layers ? a : loop_start + (a - n_layers) % (loop_end - loop_start);
    }
    bool is_loop(int layer) const { return layer >= loop_start && layer < loop_end; }
};

}  // namespace imp
