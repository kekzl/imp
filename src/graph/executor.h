#pragma once

#include "model/model.h"
#include "memory/kv_cache.h"
#include "memory/ssm_state.h"
#include "memory/layer_offload.h"
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

    // SSM state for Mamba2 layers (nullptr for non-hybrid models)
    SSMState* ssm_state = nullptr;
    int ssm_seq_id = 0;  // sequence ID for SSM state access

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
    bool init(const Model& model, DType compute_dtype = DType::FP16, bool use_pdl = false,
              int max_batch_size = 1, int max_seq_len = 0);

    // Run the full forward pass and return the sampled token ID.
    int32_t forward(const InferenceState& state, cudaStream_t stream = nullptr);

    // Batched forward: returns one sampled token per sequence.
    std::vector<int32_t> forward_batch(const InferenceState& state, cudaStream_t stream = nullptr);

    // Run the forward pass but return raw logits instead of sampling.
    // logits_out will be a view into the internal logits buffer.
    void forward_logits(const InferenceState& state, Tensor& logits_out,
                        cudaStream_t stream = nullptr);

    // Sample tokens from pre-computed logits (for use after CUDA graph execution).
    std::vector<int32_t> sample_from_logits(const Tensor& logits,
                                             const InferenceState& state,
                                             cudaStream_t stream = nullptr);

    // Async decode: runs forward pass reading token from device memory (d_token_id),
    // then samples and writes result back to d_token_id. No host-device sync.
    // h_mapped: mapped pinned memory for host-side token readback (polled async).
    // Returns immediately. Host reads *h_mapped to get the token.
    void forward_decode_async(const InferenceState& state,
                              int32_t* d_token_id, int32_t* h_mapped,
                              cudaStream_t stream = nullptr);

    // Set KV layer mapping (must be called before forward pass for hybrid models)
    void set_kv_layer_map(std::vector<int> map) { kv_layer_map_ = std::move(map); }

    // Set layer offload manager (optional, for weight offloading)
    void set_offload_manager(LayerOffloadManager* mgr) { offload_mgr_ = mgr; }

    // Resize workspace for a different max token count (Phase 4: decode-mode optimization).
    // Uses cudaFreeAsync/cudaMallocAsync for near-instant resize via CUDA memory pool.
    bool resize_workspace(int new_max_tokens, cudaStream_t stream);

    // Get a view of the logits buffer for n tokens (for CUDA graph replay,
    // where forward_logits isn't called but the graph writes to this buffer).
    Tensor get_logits_view(int n) const { return view_tokens(logits_, n); }

    // Pre-allocated device buffer for sampling output (stable address for CUDA graph).
    int32_t* d_sample_result() const { return d_sample_result_; }

private:
    const Model* model_ = nullptr;
    DType compute_dtype_ = DType::FP16;
    bool initialized_ = false;
    int max_tokens_ = 0;
    int max_logit_tokens_ = 0;  // max tokens needing LM head projection (= max_batch_size)
    int cur_n_tokens_ = 0;  // set by forward_logits for use by run_ffn

    // Programmatic Dependent Launch: when true, custom kernels have the PDL
    // attribute set so the GPU can overlap tail of one kernel with head of next.
    bool use_pdl_ = false;

    // --- Persistent GPU workspace (always valid, not reconfigured) ---
    void* persistent_workspace_ = nullptr;
    size_t persistent_workspace_size_ = 0;

    // Persistent activation tensors (views into persistent_workspace_)
    Tensor hidden_;        // [max_tokens, d_model]
    Tensor residual_;      // [max_tokens, d_model]
    Tensor norm_out_;      // [max_tokens, d_model]
    Tensor logits_;        // [max_logit_tokens, vocab_size]

    // --- Shared GPU workspace (reconfigured per layer phase) ---
    // Sized to max(attn_size, ffn_size, moe_size, ssm_size).
    // Tensor views are set up at the start of each run_* function.
    void* shared_workspace_ = nullptr;
    size_t shared_workspace_size_ = 0;
    int shared_workspace_max_tokens_ = 0;  // token count used for current allocation

    // Pre-computed phase sizes (for max_tokens_)
    size_t attn_shared_size_ = 0;
    size_t ffn_shared_size_ = 0;
    size_t moe_shared_size_ = 0;
    size_t ssm_shared_size_ = 0;

    // Attention phase tensors (views into shared_workspace_, set by configure_attn_workspace)
    Tensor q_;             // [max_tokens, n_heads * head_dim]
    Tensor k_;             // [max_tokens, n_kv_heads * head_dim]
    Tensor v_;             // [max_tokens, n_kv_heads * head_dim]
    Tensor attn_out_;      // [max_tokens, n_heads * head_dim]
    Tensor proj_out_;      // [max_tokens, d_model]

    // Dense FFN phase tensors (views into shared_workspace_, set by configure_ffn_workspace)
    Tensor gate_out_;      // [max_tokens, d_ff]
    Tensor up_out_;        // [max_tokens, d_ff]
    Tensor swiglu_out_;    // [max_tokens, d_ff]
    Tensor ffn_out_;       // [max_tokens, d_model]

    // MoE phase tensors (views into shared_workspace_, set by configure_moe_workspace)
    MoeRoutingBuffers moe_routing_buffers_;
    Tensor moe_gate_logits_;    // [max_tokens, n_experts] FP32
    Tensor moe_gathered_;       // [max_tokens * top_k, d_model] compute_dtype
    Tensor moe_expert_gate_;    // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor moe_expert_up_;      // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor moe_expert_swiglu_;  // [max_tokens * top_k, expert_d_ff] compute_dtype
    Tensor moe_expert_down_;    // [max_tokens * top_k, d_model] compute_dtype
    Tensor moe_scatter_out_;    // [max_tokens, d_model] FP32 (scatter output)

    // SSM phase tensors (views into shared_workspace_, set by configure_ssm_workspace)
    Tensor ssm_proj_buf_;   // [max_tokens, ssm_in_dim] for ssm_in projection
    Tensor ssm_xBC_buf_;    // [max_tokens, conv_channels] for conv output
    Tensor ssm_y_buf_;      // [max_tokens, inner_size] for scan output
    Tensor ssm_z_buf_;      // [max_tokens, inner_size] for gate
    Tensor ssm_out_buf_;    // [max_tokens, d_model] for ssm_out projection
    Tensor ssm_dt_buf_;     // [max_tokens, n_heads] for dt after split

    // --- Separately allocated buffers (not part of unified workspace) ---

    // On-the-fly dequant scratch buffer for quantized expert weights.
    void* moe_dequant_buf_ = nullptr;
    size_t moe_dequant_buf_size_ = 0;

    // GPU staging buffer for one expert's raw quantized bytes (H2D copy).
    void* moe_raw_staging_buf_ = nullptr;
    size_t moe_raw_staging_size_ = 0;

    // On-the-fly dequant scratch buffer for non-MoE quantized weights (Q8_0/Q6_K).
    void* dequant_scratch_ = nullptr;
    size_t dequant_scratch_size_ = 0;

    // Pre-allocated sampling result buffers (avoids cudaMalloc/cudaFree per token).
    int32_t* d_sample_result_ = nullptr;  // device buffer for argmax/sample kernel output

    // MMVQ (dp4a) scratch buffers for quantized input vector.
    // Allocated once during init, reused each layer for decode GEMV.
    void* q8_1_buf_ = nullptr;   // block_q8_1 array, size = max_dim / 32 * sizeof(block_q8_1)
    float* d8_buf_ = nullptr;    // float scale array, size = max_dim / 32 * sizeof(float)
    int q8_1_max_blocks_ = 0;    // max K/32 across all weight matrices

    // Split-K paged attention scratch buffer.
    // Holds partial softmax states: [batch * n_heads * max_splits * (2 + head_dim)] floats.
    void* splitk_scratch_ = nullptr;
    size_t splitk_scratch_size_ = 0;

    // --- Layer index mappings ---

    // Mapping from global layer index to SSM layer index (for SSMState access)
    std::vector<int> ssm_layer_map_;  // ssm_layer_map_[global_idx] = ssm_idx, or -1

    // Mapping from global layer index to KV cache layer index (for attention layers only)
    std::vector<int> kv_layer_map_;   // kv_layer_map_[global_idx] = kv_idx, or -1

    // --- Model feature flags (set during init for workspace computation) ---
    bool has_moe_ = false;
    bool has_ssm_ = false;
    bool has_dense_ffn_ = false;

    // Max expert FFN hidden dim from actual packed tensor shapes (may differ from cfg.expert_d_ff)
    int max_expert_eff_ = 0;

    // --- Layer offload manager (non-owning, set by engine) ---
    LayerOffloadManager* offload_mgr_ = nullptr;

    // --- Allocation and configuration methods ---

    void allocate_persistent_workspace(int max_tokens);
    void allocate_shared_workspace(int max_tokens);
    void allocate_auxiliary_buffers();  // dequant scratch, MoE staging, routing buffers
    void free_buffers();

    // Compute shared workspace sizes for each phase (stored in *_shared_size_ members)
    void compute_shared_sizes(int max_tokens);

    // Configure tensor views into shared_workspace_ for each phase.
    // Called at the start of each run_* function. Pure pointer arithmetic, no allocation.
    void configure_attn_workspace(int max_tokens);
    void configure_ffn_workspace(int max_tokens);
    void configure_moe_workspace(int max_tokens);
    void configure_ssm_workspace(int max_tokens);

    // Per-layer helpers
    void run_attention(int layer, const InferenceState& state, cudaStream_t stream);
    void run_ffn(int layer, cudaStream_t stream);
    void run_moe_ffn(int layer, cudaStream_t stream);
    void run_ssm(int layer, const InferenceState& state, cudaStream_t stream);

    // Layer type detection (based on tensor presence)
    bool layer_has_attention(int layer) const;
    bool layer_has_ssm(int layer) const;
    bool layer_has_moe(int layer) const;
    bool layer_has_dense_ffn(int layer) const;

    // Write computed K/V into KV cache blocks
    void write_kv_cache(int layer, const InferenceState& state, cudaStream_t stream);

    // Create a Tensor view of the first n_tokens rows of a max_tokens buffer.
    Tensor view_tokens(const Tensor& buf, int n_tokens) const;
};

} // namespace imp
