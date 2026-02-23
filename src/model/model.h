#pragma once

#include "core/tensor.h"
#include "model/model_arch.h"
#include "model/tokenizer.h"
#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>

namespace imp {

// GGML quantization type stored alongside tensor for dequant dispatch
enum class GGMLQuantType : uint32_t {
    NONE = 0,
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13,
    Q6_K = 14, Q8_K = 15, BF16 = 30,
};

struct ModelConfig {
    ModelArch arch = ModelArch::GENERIC;
    int n_layers = 0, n_heads = 0, n_kv_heads = 0;
    int d_model = 0, d_ff = 0, vocab_size = 0, max_seq_len = 0;
    int head_dim = 0;  // 0 = infer as d_model / n_heads
    float rope_theta = 10000.0f, rms_norm_eps = 1e-5f;
    int n_experts = 0, n_experts_active = 0, expert_d_ff = 0;

    // Per-layer config (empty for standard transformers)
    std::vector<int> n_kv_heads_per_layer;  // 0 = no attention this layer
    std::vector<int> d_ff_per_layer;        // 0 = no dense FFN (SSM or attention-only)

    // Mamba2 SSM config
    int ssm_conv_kernel = 0;    // 4
    int ssm_state_size = 0;     // 128
    int ssm_group_count = 0;    // 8
    int ssm_inner_size = 0;     // 4096
    int ssm_dt_rank = 0;        // 64
    int rope_dim = 0;           // 0 = full head_dim, 84 = partial

    // Extended MoE config
    int n_experts_shared = 0;       // 1
    int expert_shared_d_ff = 0;     // 3712
    float expert_weights_scale = 1.0f;  // 2.5
    bool expert_weights_norm = false;
    bool moe_sigmoid_gating = false;   // Nemotron-H uses sigmoid instead of softmax
};

struct TransformerLayer {
    Tensor wq, wk, wv, wo, attn_norm;
    Tensor attn_q_norm, attn_k_norm;  // QK-norm (Qwen3-style per-head RMSNorm)
    Tensor w_gate, w_up, w_down, ffn_norm;
    Tensor moe_gate;
    std::vector<Tensor> expert_w_gate, expert_w_up, expert_w_down;

    // Packed expert tensors (3D: [n_experts, rows, cols]) loaded from GGUF *_exps
    // These are temporary: weight_upload slices them into the per-expert vectors above.
    Tensor expert_gate_packed, expert_up_packed, expert_down_packed;
    GGMLQuantType expert_gate_qtype = GGMLQuantType::NONE;
    GGMLQuantType expert_up_qtype   = GGMLQuantType::NONE;
    GGMLQuantType expert_down_qtype = GGMLQuantType::NONE;

    // Shared expert (always-active, e.g. Nemotron/DeepSeek)
    Tensor w_up_shared, w_down_shared, w_gate_shared;
    GGMLQuantType w_up_shared_qtype   = GGMLQuantType::NONE;
    GGMLQuantType w_down_shared_qtype = GGMLQuantType::NONE;
    GGMLQuantType w_gate_shared_qtype = GGMLQuantType::NONE;

    // Per-group scales for quantized weights (GPU, FP16)
    Tensor wq_scales, wk_scales, wv_scales, wo_scales;
    Tensor w_gate_scales, w_up_scales, w_down_scales;

    // Store original GGML quant types for dequant dispatch
    GGMLQuantType wq_qtype = GGMLQuantType::NONE;
    GGMLQuantType wk_qtype = GGMLQuantType::NONE;
    GGMLQuantType wv_qtype = GGMLQuantType::NONE;
    GGMLQuantType wo_qtype = GGMLQuantType::NONE;
    GGMLQuantType w_gate_qtype = GGMLQuantType::NONE;
    GGMLQuantType w_up_qtype = GGMLQuantType::NONE;
    GGMLQuantType w_down_qtype = GGMLQuantType::NONE;

    // Mamba2 SSM weights
    Tensor ssm_in, ssm_out;              // Projections
    Tensor ssm_conv1d_w, ssm_conv1d_b;   // Conv1d weight + bias
    Tensor ssm_dt_b;                      // dt bias
    Tensor ssm_a, ssm_d;                  // A (log) and D (skip connection)
    Tensor ssm_norm_w;                    // Group RMSNorm weight
    GGMLQuantType ssm_in_qtype = GGMLQuantType::NONE;
    GGMLQuantType ssm_out_qtype = GGMLQuantType::NONE;

    // Router bias (Nemotron MoE)
    Tensor moe_router_bias;
};

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
    int n_layers() const { return static_cast<int>(layers_.size()); }

    Tokenizer* tokenizer() const { return tokenizer_.get(); }
    void set_tokenizer(std::unique_ptr<Tokenizer> tok) { tokenizer_ = std::move(tok); }

    // Upload mmap'd weights to GPU, dequantizing as needed.
    // For Q4_0: splits block format into packed nibbles + scales on GPU.
    // For Q8_0: dequantizes to FP16 on GPU.
    // For F16/BF16: direct upload.
    // For F32: converts to compute_dtype and uploads.
    bool upload_weights_gpu(DType compute_dtype = DType::FP16, cudaStream_t stream = nullptr);

    bool gpu_weights_ready() const { return gpu_weights_ready_; }

    ModelConfig config_;
    Tensor tok_emb_, out_norm_, out_proj_;
    GGMLQuantType tok_emb_qtype_ = GGMLQuantType::NONE;
    GGMLQuantType out_norm_qtype_ = GGMLQuantType::NONE;
    GGMLQuantType out_proj_qtype_ = GGMLQuantType::NONE;
    std::vector<TransformerLayer> layers_;
    std::unique_ptr<Tokenizer> tokenizer_;

    void* mmap_base_ = nullptr;
    size_t mmap_size_ = 0;

    bool gpu_weights_ready_ = false;
    std::vector<void*> gpu_allocations_;
    std::vector<void*> host_pinned_;  // mmap regions pinned via cudaHostRegister
};

} // namespace imp
