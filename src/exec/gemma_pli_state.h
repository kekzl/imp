#pragma once
// Gemma-4 E-variant executor state (executor_gemma_pli.cpp): per-layer input (PLE) scratch and,
// per KV source layer, its normed + rotated K/V rows of the current forward for the shared layers.

#include "core/tensor.h"
#include "model/model_config.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace imp {

struct GemmaPliState {
    Tensor inputs;  // [max_tokens, n_layers * pli_dim] FP16, this forward's per-layer inputs
    Tensor gate;    // [max_tokens, pli_dim] FP16 scratch
    Tensor out;     // [max_tokens, d_model] FP16 scratch
    struct KvShareRows {
        void* k = nullptr;
        void* v = nullptr;
        int64_t cols = 0;  // nkv * hd of the source layer
    };
    std::vector<KvShareRows> kv_rows;  // [n_layers], set on source layers only
};

// Source layer: copy this chunk's K/V rows (no-op on a layer no other layer reads).
void kv_share_save(const GemmaPliState& s, int layer, const Tensor& k, const Tensor& v, int n,
                   cudaStream_t stream);
// Shared layer: point k/v at its source's saved rows. false = no rows (caller must not attend).
[[nodiscard]] bool kv_share_bind(const GemmaPliState& s, const ModelConfig& cfg, int layer, Tensor& k,
                                 Tensor& v, int n);

}  // namespace imp
