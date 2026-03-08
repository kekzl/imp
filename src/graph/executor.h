#pragma once

#include "model/model.h"
#include "memory/kv_cache.h"
#include "memory/ssm_state.h"
#include "memory/layer_offload.h"
#include "compute/moe_routing.h"
#include "compute/json_constrain.h"
#include "quant/nvfp4_quant.h"
#include "compute/gemm_cutlass_sm120.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <unordered_map>

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
    float min_p = 0.0f;
    float typical_p = 1.0f;           // Locally typical sampling (1.0 = disabled)
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;

    // DRY (Don't Repeat Yourself) penalty
    float dry_multiplier = 0.0f;     // 0 = disabled
    float dry_base = 1.75f;
    int dry_allowed_length = 2;
    int dry_penalty_last_n = 0;      // 0 = full history
    const int32_t* host_penalty_tokens = nullptr;  // HOST pointer for DRY scanning

    // Mirostat v2 adaptive entropy sampling
    int mirostat = 0;             // 0=off, 2=Mirostat v2
    float mirostat_tau = 5.0f;    // Target entropy
    float mirostat_eta = 0.1f;    // Learning rate
    mutable float mirostat_mu = 0.0f;  // Running variable (updated by sampling)

    // Token history for penalty computation (device pointer, owned by engine)
    const int32_t* penalty_tokens = nullptr;
    int n_penalty_tokens = 0;

    // Logprobs: when true, forward() copies logits to h_logits_pinned_ for CPU extraction
    bool logprobs = false;
    int top_logprobs = 0;

    // JSON mode: when non-null, apply logit mask before sampling
    JsonConstrainer* json_constrainer = nullptr;

    // Vision: when non-null, replace vision_token_id positions with vision embeddings
    const half* vision_embeddings = nullptr;  // [n_vision_tokens, d_model] FP16 on device
    int vision_token_id = -1;                 // <image_soft_token> ID
    int n_vision_tokens = 0;                  // 256
};

// Imperative executor for the transformer forward pass.
//
// The Graph class provides a DAG representation for visualization and debugging,
// but this executor hardcodes the standard transformer forward pass for
// efficiency. No graph walking is done at runtime.
class GraphExecutor {
public:
    // FP8 weight cache entry (public for use by static gemm_dispatch)
    struct FP8CacheEntry {
        Tensor weight;       // [N, K] FP8_E4M3 on device
        float host_scale;    // absmax / 448
        float* d_scale;      // device-side scale (1 float, for gemm_cublaslt bScale)
    };

    GraphExecutor() = default;
    ~GraphExecutor();

    // Phase 1: Initialize model reference, compute workspace sizes, enable PDL.
    // Does NOT allocate GPU memory — call allocate_workspaces() after weight upload.
    [[nodiscard]] bool init(const Model& model, DType compute_dtype = DType::FP16, bool use_pdl = false,
                            int max_batch_size = 1, int max_seq_len = 0, bool use_fp8_prefill = false,
                            int use_nvfp4_decode = 0);

    // Phase 2: Allocate all GPU workspace buffers.
    // Call AFTER weight upload to maximize VRAM available for expert layers.
    // experts_on_host: if true, skip MoE batch dequant buffer allocation.
    [[nodiscard]] bool allocate_workspaces(bool experts_on_host = false);

    // Estimated GPU memory needed by allocate_workspaces().
    // Used by Engine to compute the expert upload reserve.
    size_t workspace_estimate() const;

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

    // Pre-dequantize quantized weights to FP16 on GPU for fast prefill GEMM.
    // Must be called AFTER model weights are uploaded to GPU.
    // cache_budget: max bytes available for weight caches (0 = unlimited).
    void pre_dequant_weights(cudaStream_t stream = nullptr, size_t cache_budget = 0);

    // Set KV layer mapping (must be called before forward pass for hybrid models)
    void set_kv_layer_map(std::vector<int> map) {
        kv_layer_map_ = std::move(map);
        // Count KV layers and initialize per-layer FP8 scale vectors
        int n_kv = 0;
        for (int idx : kv_layer_map_) {
            if (idx >= 0) n_kv = std::max(n_kv, idx + 1);
        }
        kv_scales_.assign(n_kv, 1.0f);
        kv_calibrated_.assign(n_kv, false);
    }

    // Set layer offload manager (optional, for weight offloading)
    void set_offload_manager(LayerOffloadManager* mgr) { offload_mgr_ = mgr; }

    // Resize workspace for a different max token count (Phase 4: decode-mode optimization).
    // Uses cudaFreeAsync/cudaMallocAsync for near-instant resize via CUDA memory pool.
    [[nodiscard]] bool resize_workspace(int new_max_tokens, cudaStream_t stream);

    // Get a view of the logits buffer for n tokens (for CUDA graph replay,
    // where forward_logits isn't called but the graph writes to this buffer).
    Tensor get_logits_view(int n) const { return view_tokens(logits_, n); }

    // Release the MoE batch dequant buffer when expert weights are on host.
    // Call after weight upload if experts didn't fit on GPU.
    void release_moe_batch_buf();

    // Pre-allocated device buffer for sampling output (stable address for CUDA graph).
    int32_t* d_sample_result() const { return d_sample_result_; }

    // Pinned host buffer for logprobs extraction.
    float* h_logits_pinned() const { return h_logits_pinned_; }

    // Ensure pinned logits buffer is allocated for the given vocab size.
    void ensure_logits_pinned(int vocab_size);

private:
    const Model* model_ = nullptr;
    DType compute_dtype_ = DType::FP16;
    float norm_w_off_ = 0.0f;  // Gemma: 1.0 (norms use w+1 instead of w)
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
    Tensor hidden_;        // [max_tokens, d_model] FP16
    Tensor residual_;      // [max_tokens, d_model] FP16
    Tensor norm_out_;      // [max_tokens, d_model] FP16
    Tensor logits_;        // [max_logit_tokens, vocab_size]

    // FP32 residual accumulator for post-norm architectures (Gemma-3).
    // Prevents FP16 overflow in the residual stream over many layers.
    // The FP32 tensor is the "true" hidden state; the FP16 hidden_ is only
    // used as input to RMSNorm (which is scale-invariant, so clamping is safe).
    // nullptr for pre-norm models (LLaMA, Qwen, etc.).
    void* fp32_accum_buf_ = nullptr;
    Tensor fp32_hidden_;   // [max_tokens, d_model] FP32 — true hidden state

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

    // cuBLAS attention S-matrix workspace (separately allocated, not part of shared workspace).
    // [n_heads, max_tokens, max_tokens] FP16 — used only during prefill.
    void* attn_scores_buf_ = nullptr;
    size_t attn_scores_buf_size_ = 0;
    Tensor attn_scores_;   // 3D tensor view into attn_scores_buf_

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

    // On-the-fly dequant scratch buffer for quantized expert weights (1 expert).
    void* moe_dequant_buf_ = nullptr;
    size_t moe_dequant_buf_size_ = 0;

    // Batch dequant buffer for MoE prefill: holds a chunk of experts' weights
    // dequanted to FP16. Sized for L2-resident chunked processing: dequant a
    // chunk of experts, then immediately GEMM while FP16 data is still in L2.
    void* moe_batch_dequant_buf_ = nullptr;
    size_t moe_batch_dequant_buf_size_ = 0;

    // Pre-allocated device pointer array for batched MoE GEMM (avoids per-call cudaMallocAsync).
    // Layout: [A_ptrs..., B_ptrs..., C_ptrs...] = 3 * n_experts void pointers.
    void** d_moe_work_ptrs_ = nullptr;
    int d_moe_work_ptrs_count_ = 0;  // n_experts used for allocation

    // Per-expert FP8 scale buffer: [n_experts] floats on device.
    // Used by calibrate_fp8_scales_per_expert() for dynamic per-expert scaling.
    float* d_moe_fp8_scales_ = nullptr;

    // Device-side weight pointer array for device-grouped GEMM.
    // Eliminates host sync by keeping weight pointers on GPU.
    // Stored as void** for easy cudaMemcpy; cast to const void** at call sites.
    void** d_moe_weight_ptrs_ = nullptr;
    int d_moe_weight_ptrs_count_ = 0;

    // GPU staging buffer for one expert's raw quantized bytes (H2D copy).
    void* moe_raw_staging_buf_ = nullptr;
    size_t moe_raw_staging_size_ = 0;

    // On-the-fly dequant scratch buffer for non-MoE quantized weights (Q8_0/Q6_K).
    void* dequant_scratch_ = nullptr;
    size_t dequant_scratch_size_ = 0;

    // Pre-dequantized FP16 weight cache for prefill GEMM (avoids per-layer dequant overhead).
    // Maps raw quantized weight pointer -> pre-dequanted FP16 Tensor on GPU.
    // Populated at init time; decode still uses dp4a GEMV on raw quantized weights.
    std::unordered_map<const void*, Tensor> fp16_cache_;
    size_t fp16_cache_bytes_ = 0;  // total VRAM used by FP16 cache

    // FP8 E4M3 weight cache for prefill GEMM (2x compute throughput on sm_120).
    // When use_fp8_cache_ is set, weights are cached as FP8 instead of FP16.
    std::unordered_map<const void*, FP8CacheEntry> fp8_cache_;
    size_t fp8_cache_bytes_ = 0;
    bool use_fp8_cache_ = false;

    // Scratch buffers for FP8 activation quantization (allocated once, reused per GEMM)
    void* fp8_act_buf_ = nullptr;       // max_tokens * max_dim bytes
    size_t fp8_act_buf_size_ = 0;
    float* d_act_scale_ = nullptr;      // 1 float on device

    // NVFP4 decode weight cache: quantized from FP16 cache at init.
    // Provides 31-47% bandwidth reduction vs raw Q8_0/Q6_K during decode GEMV.
    // Mode: 0=off, 1=additive (FP16 + NVFP4), 2=only (NVFP4 replaces FP16)
    std::unordered_map<const void*, NvFP4QuantResult> nvfp4_cache_;
    size_t nvfp4_cache_bytes_ = 0;
    int use_nvfp4_decode_ = 0;

    // NVFP4 MoE expert weight cache: per-expert NVFP4 quantization.
    // Keyed by packed expert tensor data pointer (expert_gate_packed.data etc.)
    std::unordered_map<const void*, NvFP4MoEQuantResult> nvfp4_moe_cache_;
    size_t nvfp4_moe_cache_bytes_ = 0;

    // CUTLASS sm_120 block-scaled NVFP4 weight cache for native FP4 prefill GEMM.
    // Keyed by original weight data pointer (same as nvfp4_cache_).
    // Populated after NVFP4 quantization if sm_120 is available.
    std::unordered_map<const void*, CutlassNvFP4Weight> cutlass_nvfp4_cache_;
    size_t cutlass_nvfp4_cache_bytes_ = 0;

    // Pre-allocated activation buffers for CUTLASS NVFP4 prefill.
    void* cutlass_act_data_ = nullptr;     // [max_tokens, max_K/2] packed FP4
    void* cutlass_act_sf_ = nullptr;       // SfAtom scale factors
    size_t cutlass_act_data_size_ = 0;
    size_t cutlass_act_sf_size_ = 0;
    void* cutlass_workspace_ = nullptr;    // CUTLASS GEMM workspace
    size_t cutlass_workspace_size_ = 0;

    // Fused KV weight cache: concatenated [wk; wv] as [2*nkv*hd, d_model] FP16.
    // Enables strided batched GEMM for K+V in a single cuBLAS call during prefill.
    // Key = layer index. Only populated for layers where both wk/wv are FP16-cached.
    std::unordered_map<int, Tensor> fused_kv_cache_;

    // Fused gate+up weight cache: concatenated [w_gate; w_up] as [2*d_ff, d_model] FP16.
    // Enables strided batched GEMM for gate+up in a single cuBLAS call during prefill.
    // Key = layer index. Only populated for layers where both w_gate/w_up are FP16-cached.
    std::unordered_map<int, Tensor> fused_gate_up_cache_;

    // Pre-allocated sampling result buffers (avoids cudaMalloc/cudaFree per token).
    int32_t* d_sample_result_ = nullptr;  // device buffer for argmax/sample kernel output

    // Pinned host buffer for logprobs extraction (D2H copy of logits)
    float* h_logits_pinned_ = nullptr;  // [vocab_size] pinned host memory
    int h_logits_pinned_size_ = 0;      // vocab_size used for allocation

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

    // Per-KV-layer FP8 scales for online calibration.
    // Scale = absmax / 448.0; used as inv_scale = 1/scale for write, scale for read.
    std::vector<float> kv_scales_;       // [n_kv_layers] per-layer FP8 scale
    std::vector<bool>  kv_calibrated_;   // [n_kv_layers] whether scale has been calibrated

    // YaRN correction dimension boundaries [2], precomputed at init.
    // yarn_corr_dims_[0] = start (full interpolation below), yarn_corr_dims_[1] = end (full extrapolation above)
    float yarn_corr_dims_[2] = {0.0f, 0.0f};

    // LongRoPE pre-computed inverse frequencies (device memory)
    float* longrope_short_freqs_ = nullptr;  // [rope_pairs] device
    float* longrope_long_freqs_  = nullptr;  // [rope_pairs] device
    int    longrope_orig_max_pos_ = 0;
    int    longrope_n_pairs_ = 0;

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
    void allocate_auxiliary_buffers(bool skip_batch_dequant = false);  // dequant scratch, MoE staging, routing buffers
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
