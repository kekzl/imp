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
    void set_name(const std::string& n) { name_ = n; }

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
