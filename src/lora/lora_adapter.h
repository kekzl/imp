#pragma once

// LoRA adapter (PEFT format) for runtime low-rank deltas — issue #522.
//
// Design: NO weight patching. The base weights (FP16 cache / NVFP4 cache /
// raw GGUF dp4a) stay untouched; the adapter contributes an activation-path
// delta  y += (alpha/r) * (x · A^T) · B^T  per adapted projection. That makes
// the feature quant-path-agnostic by construction and hot-swap = swapping an
// adapter pointer (plus a decode-graph re-capture, handled by the engine).
//
// Loads HuggingFace PEFT directories:
//   adapter_config.json          (r, lora_alpha, use_rslora, target_modules)
//   adapter_model.safetensors    (base_model.model.model.layers.N.<proj>.lora_{A,B}.weight)
// A is [r, K], B is [N, r]; F32/F16/BF16 accepted, stored as F16 on device.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace imp {

enum class LoraProj : uint8_t { Q = 0, K, V, O, GATE, UP, DOWN, COUNT };

struct LoraWeights {
    void* A = nullptr;  // device F16 [r, K]
    void* B = nullptr;  // device F16 [N, r]
    int r = 0;
    int K = 0;
    int N = 0;
};

// Base-model widths an adapter's declared shapes are held against at load
// (AUDIT_arch_2026 F1-6): K is read from the projection input, N is written
// to its output, so a wrong pair is an out-of-bounds kernel, not a bad answer.
struct LoraDims {
    int d_model = 0;  // input of q/k/v/gate/up, output of o/down
    int q_out = 0;    // n_heads * head_dim
    int kv_out = 0;   // n_kv_heads * head_dim
    int d_ff = 0;     // dense FFN width; 0 = the model has no dense FFN
};

// Expected [K, N] of one projection's adapter; false when the model has no
// such projection (an FFN target on a model without a dense FFN).
inline bool lora_proj_expected(LoraProj p, const LoraDims& d, int* K, int* N) {
    switch (p) {
        case LoraProj::Q:
            *K = d.d_model, *N = d.q_out;
            return true;
        case LoraProj::K:
        case LoraProj::V:
            *K = d.d_model, *N = d.kv_out;
            return true;
        case LoraProj::O:
            *K = d.q_out, *N = d.d_model;
            return true;
        case LoraProj::GATE:
        case LoraProj::UP:
            *K = d.d_model, *N = d.d_ff;
            return d.d_ff > 0;
        case LoraProj::DOWN:
            *K = d.d_ff, *N = d.d_model;
            return d.d_ff > 0;
        default:
            return false;
    }
}

class LoraAdapter {
public:
    LoraAdapter() = default;
    ~LoraAdapter();
    LoraAdapter(const LoraAdapter&) = delete;
    LoraAdapter& operator=(const LoraAdapter&) = delete;

    // Load a PEFT adapter directory (or a bare .safetensors file, in which
    // case alpha/r fall back to the tensor shapes with scale=1). Returns
    // false with a logged reason on any parse/shape problem.
    bool load(const std::string& path, int n_layers);
    // Every loaded pair must match the base model's widths; `why` names the
    // first mismatch. Call after load(), before the adapter can be selected.
    bool check_dims(const LoraDims& d, std::string* why) const;

    const LoraWeights* get(int layer, LoraProj p) const {
        if (layer < 0 || layer >= static_cast<int>(layers_.size()))
            return nullptr;
        const LoraWeights& w = layers_[layer].proj[static_cast<int>(p)];
        return (w.A && w.B) ? &w : nullptr;
    }
    bool has_qkv(int layer) const {
        return get(layer, LoraProj::Q) || get(layer, LoraProj::K) || get(layer, LoraProj::V);
    }
    bool has_any() const { return n_tensors_ > 0; }
    float scale() const { return scale_; }
    int max_rank() const { return max_rank_; }
    const std::string& name() const { return name_; }

private:
    struct LayerSlots {
        LoraWeights proj[std::to_underlying(LoraProj::COUNT)];
    };
    std::vector<LayerSlots> layers_;
    std::vector<void*> device_allocs_;
    float scale_ = 1.0f;
    int max_rank_ = 0;
    int n_tensors_ = 0;
    std::string name_;
};

}  // namespace imp
