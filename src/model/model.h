#pragma once

#include "model/hf_config_loader.h"
#include "model/model_config.h"
#include "model/tokenizer.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace imp {

class Model {
public:
    Model() = default;
    ~Model();

    const ModelConfig& config() const { return config_; }

    // Sampling/EOS defaults shipped by the model author in
    // generation_config.json (SafeTensors only; empty for GGUF). Sentinel
    // values (<0) mean the field was not present.
    const HFConfigLoader::GenerationConfig& generation_config() const {
        return generation_config_;
    }
    const TransformerLayer& layer(int i) const { return layers_[i]; }
    TransformerLayer& layer(int i) { return layers_[i]; }
    const Tensor& token_embedding() const { return tok_emb_; }
    const Tensor& output_norm() const { return out_norm_; }
    const Tensor& output_proj() const { return out_proj_; }
    int n_layers() const { return static_cast<int>(layers_.size()); }

    Tokenizer* tokenizer() const { return tokenizer_.get(); }
    void set_tokenizer(std::unique_ptr<Tokenizer> tok) { tokenizer_ = std::move(tok); }

    // Upload mmap'd weights to GPU, dequantizing as needed.
    // For Q4_0: splits block format into packed nibbles + scales on GPU.
    // For Q8_0: dequantizes to FP16 on GPU.
    // For F16/BF16: direct upload.
    // For F32: converts to compute_dtype and uploads.
    bool upload_weights_gpu(QType compute_dtype = QType::F16, cudaStream_t stream = nullptr,
                            size_t expert_reserve_bytes = 1ULL << 30);

    bool gpu_weights_ready() const { return gpu_weights_ready_; }

    // Release a specific GPU allocation (removes from gpu_allocations_ and calls cudaFree).
    // Used when NVFP4 MoE replaces Q6K expert data to reclaim VRAM.
    void release_gpu_allocation(void* ptr);

    // Estimate total raw bytes for all expert packed tensors (for VRAM budget decisions).
    size_t estimate_expert_bytes() const;

    ModelConfig config_;
    HFConfigLoader::GenerationConfig generation_config_;
    Tensor tok_emb_, out_norm_, out_proj_;
    // (qtype mirrors removed in Stage G — read tok_emb_.qtype directly.)
    TensorID out_proj_id = kInvalidTensorID;  // registry handle for LM head (Task 3.5)
    TensorID tok_emb_id  = kInvalidTensorID;  // registry handle for token embedding
    std::vector<TransformerLayer> layers_;
    std::unique_ptr<Tokenizer> tokenizer_;

    // Load-time scratch for NVFP4 prequant scale tensors.
    // Keys:
    //   "L{idx}.{slot}"          per-layer dense (e.g. "L5.wq", "L5.w_gate_shared")
    //   "L{idx}.expert_w_{kind}.{e}"  per-expert (e.g. "L5.expert_w_gate.7")
    //   "out_proj"               LM head
    // Populated by safetensors_loader → weight_map.cpp on the SafeTensors
    // NVFP4-prequant load path. Cleared after executor_pre_dequant.cu's
    // Phase 0 promote() copies the device-side scale pointers and the FP32
    // tensor scalar onto each main weight tensor's .scales / .tensor_scale
    // sidecars. Empty for GGUF and non-NVFP4 SafeTensors models.
    std::unordered_map<std::string, NvFP4PreQuantWeight> nvfp4_scratch_;

    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    std::vector<std::pair<void*, size_t>> split_mmaps_;  // additional shard mmaps

    bool gpu_weights_ready_ = false;
    std::vector<void*> gpu_allocations_;
    std::vector<void*> host_pinned_;        // mmap regions pinned via cudaHostRegister
    std::vector<void*> host_pinned_allocs_; // cudaHostAlloc'd expert buffers (WSL2 DMA path)
    // Heap-allocated buffers used to hold permuted weight copies (e.g. Qwen3.5/3.6
    // GDN head reordering grouped→tiled). Freed with std::free() in destructor.
    std::vector<void*> host_owned_buffers_;
};

} // namespace imp
