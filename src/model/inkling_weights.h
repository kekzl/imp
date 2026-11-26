#pragma once
// Inkling2-Loop checkpoint on the device, under the vendor tensor names (model.llm.*).

#include "memory/vram_owned.h"
#include "model/inkling_config.h"
#include "quant/nvfp4_quant.h"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace imp {

class VRAMAllocator;

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
    NvFP4MoEQuantResult w13, w2;                 // NVFP4: [E, 2I, H] gate/up rows interleaved, [E, H, I]
    uint8_t *w13_q = nullptr, *w13_s = nullptr;  // MXFP8 values and per-32 exponents
    uint8_t *w2_q = nullptr, *w2_s = nullptr;
    // MXFP8, per expert: exponent bias and 2^(bias - 127) (inkling::mxfp8_expert_bias).
    int32_t *w13_bias = nullptr, *w2_bias = nullptr;
    float *w13_ts = nullptr, *w2_ts = nullptr;
};

struct InklingLayer {
    half *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *gate_attn = nullptr, *wr = nullptr;
    half* sink = nullptr;
    int proj_rows = 0;  // wq .. v taps are consecutive rows of one block starting at wq
    float *q_scale = nullptr, *rel_scale = nullptr;
    float* rel_proj = nullptr;  // [rel_extent, d_rel] (transposed from the checkpoint)
    float *norm_q = nullptr, *norm_k = nullptr, *norm_v = nullptr, *mlp_norm = nullptr;
    InklingConv k_conv, v_conv;
    float *attn_conv = nullptr, *mlp_conv = nullptr;
    float *router_w = nullptr, *router_b = nullptr;
    InklingGate gates[4];         // q, k, v, mlp (loop layers)
    float* attn_query = nullptr;  // [1 or 3, hidden]; null on layer 0
    float* mlp_query = nullptr;   // [hidden]
    InklingExperts ex;
};

struct InklingWeights;
// The dense projections (per layer the q / k / v block and the output projection) as I8B32
// (compute/gemv_i8b32.h) for calls of up to 8 rows: half the bytes per decode step.
struct InklingDense8 {
    struct Matrix {
        const half* base = nullptr;  // the fp16 rows this stands for
        size_t rows = 0;
        int K = 0;
        int8_t* q = nullptr;
        half* scales = nullptr;
    };
    std::vector<Matrix> m;
    VRAMAllocator* va = nullptr;
    std::vector<VramOwned<std::byte>> owned;
    InklingDense8() = default;
    InklingDense8(const InklingDense8&) = delete;
    InklingDense8& operator=(const InklingDense8&) = delete;
    ~InklingDense8();
    // The matrix that holds the rows starting at w, and w's first row in it; null when there is none.
    [[nodiscard]] const Matrix* find(const half* w, int K, size_t& row) const;
};
// Quantizes w's dense projections; false (out left empty) when an allocation fails.
[[nodiscard]] bool build_inkling_dense8(const InklingWeights& w, VRAMAllocator& va, InklingDense8& out);

struct InklingWeights {
    InklingConfig cfg;
    std::vector<InklingLayer> layers;
    half *embed = nullptr, *unembed = nullptr;
    float *embed_norm = nullptr, *norm = nullptr, *out_query = nullptr;
    float* msg_query = nullptr;   // [loop layers, 4 (q, k, v, mlp), hidden]
    std::vector<VramOwned<std::byte>> owned;  // every device allocation above
    VRAMAllocator* va = nullptr;              // the allocator of `owned`
    size_t device_bytes = 0;

    InklingWeights() = default;
    InklingWeights(const InklingWeights&) = delete;
    InklingWeights& operator=(const InklingWeights&) = delete;
    ~InklingWeights();
};

[[nodiscard]] bool parse_inkling_config(const std::string& dir, InklingConfig& cfg);
// Uploads the text model from <dir>/model-*.safetensors. False = a tensor is missing or mis-shaped.
// with_vocab false: embedding and unembedding stay with the caller (the engine's standard path).
// Allocates from va under the tag inkling_weights.
[[nodiscard]] bool load_inkling_weights(const std::string& dir, InklingWeights& w, VRAMAllocator& va,
                                        bool with_vocab = true);

}  // namespace imp
