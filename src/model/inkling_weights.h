#pragma once
// Inkling2-Loop checkpoint on the device, under the vendor tensor names (model.llm.*).

#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>

#include <cstdint>
#include <string>
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
    bool is_loop(int layer) const { return layer >= loop_start && layer < loop_end; }
};

struct InklingConv {
    float* w = nullptr;         // static: [C, 4], tap 3 = current token
    float* static_w = nullptr;  // dynamic: [4, C], tap 0 = current token
    half* proj = nullptr;       // dynamic: [n_kv_heads * 4, hidden]
};

struct InklingGate {
    half* down = nullptr;    // [rank, 2 * hidden]
    half* forget = nullptr;  // [hidden, rank]
    half* gain = nullptr;    // [hidden, rank]
};

struct InklingExperts {
    bool fp8 = false;
    NvFP4MoEQuantResult w13, w2;                  // NVFP4: [E, 2I, H] gate/up rows interleaved, [E, H, I]
    uint8_t *w13_q = nullptr, *w13_s = nullptr;   // MXFP8 values and per-32 exponents
    uint8_t *w2_q = nullptr, *w2_s = nullptr;
};

struct InklingLayer {
    half *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *gate_attn = nullptr, *wr = nullptr;
    half* sink = nullptr;
    float *q_scale = nullptr, *rel_scale = nullptr, *rel_proj = nullptr;
    float *norm_q = nullptr, *norm_k = nullptr, *norm_v = nullptr, *mlp_norm = nullptr;
    InklingConv k_conv, v_conv;
    float *attn_conv = nullptr, *mlp_conv = nullptr;
    float *router_w = nullptr, *router_b = nullptr;
    InklingGate gates[4];          // q, k, v, mlp (loop layers)
    float* attn_query = nullptr;   // [1 or 3, hidden]; null on layer 0
    float* mlp_query = nullptr;    // [hidden]
    InklingExperts ex;
};

struct InklingWeights {
    InklingConfig cfg;
    std::vector<InklingLayer> layers;
    half *embed = nullptr, *unembed = nullptr;
    float *embed_norm = nullptr, *norm = nullptr, *out_query = nullptr;
    float* msg_query = nullptr;  // [loop layers, 4 (q, k, v, mlp), hidden]
    std::vector<void*> owned;    // every device allocation above
    size_t device_bytes = 0;

    InklingWeights() = default;
    InklingWeights(const InklingWeights&) = delete;
    InklingWeights& operator=(const InklingWeights&) = delete;
    ~InklingWeights();
};

[[nodiscard]] bool parse_inkling_config(const std::string& dir, InklingConfig& cfg);
// Uploads the text model from <dir>/model-*.safetensors. False = a tensor is missing or mis-shaped.
// with_vocab false: embedding and unembedding stay with the caller (the engine's standard path).
[[nodiscard]] bool load_inkling_weights(const std::string& dir, InklingWeights& w, bool with_vocab = true);

}  // namespace imp
