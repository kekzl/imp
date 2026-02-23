#pragma once

#include "model/model.h"
#include "memory/kv_cache.h"
#include "compute/moe_routing.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <vector>

namespace imp {

// All the state needed for a single forward pass invocation.
struct InferenceState {
    // Input tokens
    const int32_t* token_ids = nullptr;   // [n_tokens] on device
    const int* positions = nullptr;        // [n_tokens] on device
    int n_tokens = 0;

    // KV cache for paged attention (decode)
    KVCache* kv_cache = nullptr;
    const int* block_tables = nullptr;     // [n_sequences, max_blocks_per_seq] on device (2D padded)
    const int* context_lens = nullptr;     // [n_sequences] on device
    int max_context_len = 0;

    // Batching
    int n_sequences = 1;                   // number of sequences in the batch
    int max_blocks_per_seq = 0;            // max blocks per sequence (for 2D block_table indexing)
    const int* seq_offsets = nullptr;      // [n_sequences+1] for ragged prefill token offsets (optional, nullptr for decode)

    // Mode
    bool is_prefill = true;

    // Sampling parameters
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = -1;
};

// Imperative executor for the transformer forward pass.
//
// The Graph class provides a DAG representation for visualization and debugging,
// but this executor hardcodes the standard transformer forward pass for
// efficiency. No graph walking is done at runtime.
class GraphExecutor {
public:
    GraphExecutor() = default;
    ~GraphExecutor();

    // Initialize buffers for the given model. compute_dtype controls the
    // dtype of intermediate activations (FP16 or BF16).
    // use_pdl: if true, enable Programmatic Dependent Launch on custom kernels
    // to reduce inter-kernel gaps in the forward pass.
    bool init(const Model& model, DType compute_dtype = DType::FP16, bool use_pdl = false);

    // Run the full forward pass and return the sampled token ID.
    int32_t forward(const InferenceState& state, cudaStream_t stream = nullptr);

    // Batched forward: returns one sampled token per sequence.
    std::vector<int32_t> forward_batch(const InferenceState& state, cudaStream_t stream = nullptr);

    // Run the forward pass but return raw logits instead of sampling.
    // logits_out will be a view into the internal logits buffer.
    void forward_logits(const InferenceState& state, Tensor& logits_out,
                        cudaStream_t stream = nullptr);

private:
    const Model* model_ = nullptr;
    DType compute_dtype_ = DType::FP16;
    bool initialized_ = false;
    int max_tokens_ = 0;
    int cur_n_tokens_ = 0;  // set by forward_logits for use by run_ffn

    // Programmatic Dependent Launch: when true, custom kernels have the PDL
    // attribute set so the GPU can overlap tail of one kernel with head of next.
    bool use_pdl_ = false;

    // Single contiguous GPU workspace; individual tensors are views into it.
    void* workspace_ = nullptr;
    size_t workspace_size_ = 0;

    // Intermediate activation tensors (views into workspace_)
    Tensor hidden_;        // [max_tokens, d_model]
    Tensor residual_;      // [max_tokens, d_model]
    Tensor norm_out_;      // [max_tokens, d_model]
    Tensor q_;             // [max_tokens, n_heads * head_dim]
    Tensor k_;             // [max_tokens, n_kv_heads * head_dim]
    Tensor v_;             // [max_tokens, n_kv_heads * head_dim]
    Tensor attn_out_;      // [max_tokens, n_heads * head_dim]
    Tensor proj_out_;      // [max_tokens, d_model]
    Tensor gate_out_;      // [max_tokens, d_ff]
    Tensor up_out_;        // [max_tokens, d_ff]
    Tensor swiglu_out_;    // [max_tokens, d_ff]
    Tensor ffn_out_;       // [max_tokens, d_model]
    Tensor logits_;        // [max_tokens, vocab_size]

    // MoE workspace (only allocated when model has MoE layers)
    MoeRoutingBuffers moe_routing_buffers_;
    void* moe_workspace_ = nullptr;
    size_t moe_workspace_size_ = 0;
    Tensor moe_gate_logits_;    // [max_tokens, n_experts] FP32
    Tensor moe_gathered_;       // [max_tokens * top_k, d_model] compute_dtype
    Tensor moe_expert_gate_;    // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor moe_expert_up_;      // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor moe_expert_swiglu_;  // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor moe_expert_down_;    // [max_tokens * top_k, d_model] compute_dtype
    Tensor moe_scatter_out_;    // [max_tokens, d_model] FP32 (scatter output)

    void allocate_buffers(int max_tokens);
    void allocate_moe_buffers(int max_tokens);
    void free_buffers();
    void free_moe_buffers();

    // Per-layer helpers
    void run_attention(int layer, const InferenceState& state, cudaStream_t stream);
    void run_ffn(int layer, cudaStream_t stream);
    void run_moe_ffn(int layer, cudaStream_t stream);

    // Write computed K/V into KV cache blocks
    void write_kv_cache(int layer, const InferenceState& state, cudaStream_t stream);

    // Create a Tensor view of the first n_tokens rows of a max_tokens buffer.
    Tensor view_tokens(const Tensor& buf, int n_tokens) const;
};

} // namespace imp
