#pragma once

#include "model/model_config.h"
#include "model/tokenizer.h"
#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace imp {

class Model {
public:
    Model() = default;
    ~Model();

    const ModelConfig& config() const { return config_; }
    const TransformerLayer& layer(int i) const { return layers_[i]; }
    TransformerLayer& layer(int i) { return layers_[i]; }
    const Tensor& token_embedding() const { return tok_emb_; }
    const Tensor& output_norm() const { return out_norm_; }
    const Tensor& output_proj() const { return out_proj_; }
    const TransformerLayer::NvFP4PreQuantWeight& nvfp4_out_proj() const { return nvfp4_out_proj_; }
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
    Tensor tok_emb_, out_norm_, out_proj_;
    // (qtype mirrors removed in Stage G — read tok_emb_.qtype directly.)
    TransformerLayer::NvFP4PreQuantWeight nvfp4_out_proj_;  // prequant LM head scales
    TensorID out_proj_id = kInvalidTensorID;  // registry handle for LM head (Task 3.5)
    TensorID tok_emb_id  = kInvalidTensorID;  // registry handle for token embedding
    std::vector<TransformerLayer> layers_;
    std::unique_ptr<Tokenizer> tokenizer_;

    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;
    std::vector<std::pair<void*, size_t>> split_mmaps_;  // additional shard mmaps

    bool gpu_weights_ready_ = false;
    std::vector<void*> gpu_allocations_;
    std::vector<void*> host_pinned_;        // mmap regions pinned via cudaHostRegister
    std::vector<void*> host_pinned_allocs_; // cudaHostAlloc'd expert buffers (WSL2 DMA path)
};

} // namespace imp
