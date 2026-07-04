#pragma once

#include "model/model.h"
#include "memory/kv_cache.h"
#include "memory/ssm_state.h"
#include "memory/layer_offload.h"
#include "compute/moe_routing.h"
#include "compute/json_constrain.h"
#include "compute/schema_constrain.h"
#include "quant/nvfp4_quant.h"
// Note: quant/turboquant.h removed (TurboQuant retired Phase 5, 2026-05-17).
#include "compute/gemm_cutlass_sm120.h"
#include "compute/gemm_cutlass_mxfp4_sm120.h"
#include "core/tensor.h"
#include "exec/expert_cache.h"
#include "exec/inference_state.h"
#include "exec/moe_ffn_context.h"
#include "exec/weight_caches.h"
#include "exec/weight_handle.h"
#include "exec/moe_workspace.h"
#include "exec/quant_scratch.h"
#include "exec/quant_pipeline.h"
#include "exec/workspace.h"
#include "runtime/storage_planner.h"
#include "runtime/vram_budget.h"
#include "runtime/config.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <functional>
#include <vector>
#include <unordered_map>
#include <utility>
#include <list>

namespace imp {

class LoraAdapter;
struct LoraWeights;

// Nvfp4DecodeContext moved to exec/quant_pipeline.h (build-only; consumed by
// the QuantPipeline phase-3 helpers).

// Imperative executor for the transformer forward pass.
//
// The Graph class provides a DAG representation for visualization and debugging,
// but this executor hardcodes the standard transformer forward pass for
// efficiency. No graph walking is done at runtime.
struct GemmContext;  // defined in gemm_context.h
class GraphExecutor {
public:
    GraphExecutor() = default;
    ~GraphExecutor();

    // Phase 1: Initialize model reference, compute workspace sizes, enable PDL.
    // Does NOT allocate GPU memory — call allocate_workspaces() after weight upload.
    [[nodiscard]] bool init(const Model& model, QType compute_dtype = QType::F16, bool use_pdl = false,
                            int max_batch_size = 1, int max_seq_len = 0, bool use_fp8_prefill = false,
                            int use_nvfp4_decode = 0, bool use_mxfp4_prefill = false);

    // Disable FP8 weight cache (must be called before pre_dequant_weights).
    void disable_fp8_prefill() {
        wcache_.use_fp8 = false;
        hints_.prefer_fp8 = false;
    }

    // Enable dual-path quantization: attention weights stay FP8, FFN weights get NVFP4.
    // Must be called before pre_dequant_weights().
    void set_dual_path_quant(bool enable) {
        wcache_.dual_path_quant = enable;
        hints_.dual_path_attn_fp8_ffn_nvfp4 = enable;
    }

    // Phase 2: Allocate all GPU workspace buffers.
    // Call AFTER weight upload to maximize VRAM available for expert layers.
    // experts_on_host: if true, skip MoE batch dequant buffer allocation.
    [[nodiscard]] bool allocate_workspaces(bool experts_on_host = false);

    // Estimated GPU memory needed by allocate_workspaces().
    // Used by Engine to compute the expert upload reserve.
    size_t workspace_estimate() const { return ws_.workspace_estimate(); }

    // Run the full forward pass and return the sampled token ID.
    int32_t forward(const InferenceState& state, cudaStream_t stream = nullptr);

    // Batched forward: returns one sampled token per sequence.
    std::vector<int32_t> forward_batch(const InferenceState& state, cudaStream_t stream = nullptr);

    // Run the forward pass but return raw logits instead of sampling.
    // logits_out will be a view into the internal logits buffer.
    void forward_logits(const InferenceState& state, Tensor& logits_out, cudaStream_t stream = nullptr);

    // Teacher-forced perplexity over tokens[0..n-1]. Call AFTER a SINGLE-CHUNK
    // prefill of the same tokens (uses the persistent-workspace hidden_, which
    // holds all-position final hidden after forward_logits). Applies the
    // tier-aware LM head to every position in chunks and returns exp(mean NLL
    // of next tokens). Bench/eval only. For chunked prefill use the
    // Engine::begin/end_perplexity_capture flow, which accumulates via
    // perplexity_nll_partial after every chunk's forward.
    double perplexity_nll(const int32_t* tokens, int n, cudaStream_t stream = nullptr);

    // Per-chunk NLL accumulation (chunked-prefill-aware imp_perplexity).
    // Applies the tier-aware LM head to hidden_[0..chunk_len-1] — the chunk
    // that prefill just forwarded, absolute corpus positions chunk_start..
    // chunk_start+chunk_len-1 — and writes -log p(next token) into
    // d_nll[global_pos]. d_tokens / d_nll are device buffers of length
    // n_total owned by the caller (Engine ppl capture). Enqueues on `stream`
    // only; the caller reduces after a sync. Overwrites the logits_
    // workspace — call only after all reads of the chunk's logits are done.
    // Optional d_match[global_pos] ∈ {0,1}: greedy argmax == actual next
    // token (production tie-break) — teacher-forced greedy-agreement probe.
    void perplexity_nll_partial(const int32_t* d_tokens, int n_total, int chunk_start,
                                int chunk_len, double* d_nll, cudaStream_t stream,
                                int32_t* d_match = nullptr);

    // Greedy verify for n-gram speculative decoding: applies the tier-aware
    // LM head to hidden_[0..n_rows-1] (the verify chunk that forward_logits
    // just ran) and writes argmax token ids to d_out[0..n_rows-1]. Same
    // logits as production greedy sampling (incl. final softcap). Overwrites
    // the logits_ workspace. Enqueues on `stream` only.
    //
    // Optional penalties (exact production parity with apply_penalties):
    // d_hist[0..n_hist) is the request's output-token history (ends with the
    // chunk's first token t0); d_draft points at the chunk tokens AFTER t0,
    // so row j additionally penalizes d_draft[0..j-1] — the same set the
    // eager path would have accumulated by that step. repeat_last_n != 0 is
    // the caller's responsibility to gate (window semantics not replicated).
    void greedy_argmax_all(int n_rows, int32_t* d_out, cudaStream_t stream,
                           const int32_t* d_hist = nullptr, int n_hist = 0,
                           const int32_t* d_draft = nullptr, float rep_pen = 1.0f,
                           float freq_pen = 0.0f, float pres_pen = 0.0f);

    // Materialize the LM-head projection of hidden_[0..n_rows) into d_out
    // ([n_rows, vocab_size] fp32) — same batched re-projection as
    // greedy_argmax_all, but copying the rows out instead of reducing them.
    // The constrained-pipeline jump-ahead (#844) consumes them across later
    // ticks, one masked sample per row. Call while the workspace that ran
    // the chunk forward is still active. Enqueues on `stream` only.
    void project_logits_all(int n_rows, float* d_out, cudaStream_t stream);

    // Sample tokens from pre-computed logits (for use after CUDA graph execution).
    std::vector<int32_t> sample_from_logits(const Tensor& logits, const InferenceState& state,
                                            cudaStream_t stream = nullptr);

    // Single-token sampling: returns one int32_t directly (avoids vector alloc).
    // Use for single-sequence decode where only one token is sampled.
    int32_t sample_single_from_logits(const Tensor& logits, const InferenceState& state,
                                      cudaStream_t stream = nullptr);

    // Async decode: runs forward pass reading token from device memory (d_token_id),
    // then samples and writes result back to d_token_id. No host-device sync.
    // h_mapped: mapped pinned memory for host-side token readback (polled async).
    // Returns immediately. Host reads *h_mapped to get the token.
    void forward_decode_async(const InferenceState& state, int32_t* d_token_id, int32_t* h_mapped,
                              cudaStream_t stream = nullptr);

    // Constrained async sampling (pipelined constrained decode): applies
    // device-side banned-token masking + the active json/schema constraint
    // mask, then samples on device. Writes the token to d_result (which must
    // be SAMPLE_SCRATCH_BYTES — multi-block sampler scratch follows the token)
    // and async-copies it to h_pinned. No host-device sync — order via an event.
    void masked_sample_async(const InferenceState& state, const Tensor& logits, int32_t* d_result,
                             int32_t* h_pinned, cudaStream_t stream);

    // Set centralized VRAM allocator for budget-tracked allocations.
    // Must be called before allocate_workspaces() / pre_dequant_weights().
    void set_vram_allocator(class VRAMAllocator* alloc) { vram_alloc_ = alloc; }

    // Pre-dequantize quantized weights to FP16 on GPU for fast prefill GEMM.
    // Must be called AFTER model weights are uploaded to GPU.
    // budget: VRAM budget with per-phase caps computed by Engine::plan_vram_budget().
    void pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget);

    // Allocate the gemm_nvfp4 dequant workspace based on max NVFP4 weight size.
    // Must be called AFTER pre_dequant_weights() so wcache_.nvfp4 is populated.
    // Skips allocation if no NVFP4 weights exist or if the largest weight
    // exceeds a sanity cap (currently 512 MiB) — in those cases the gemm_nvfp4
    // fallback continues to use lazy cudaMalloc on non-captured streams.
    bool allocate_nvfp4_dequant_workspace();

    // Build a CUTLASS NVFP4 weight for the LM head so batched decode (n>1) can
    // do a single tensor-core GEMM (weight read once) instead of the per-row /
    // batched-M GEMV (weight read ceil(M/4)x). Only the SfAtom scale buffer is
    // allocated (~vocab*d_model/16 bytes); the FP4 data is borrowed from the
    // NVFP4 decode cache. No-op unless serving (max_logit_tokens_ > 1) and the
    // LM head is NVFP4. Must run AFTER pre_dequant_weights().
    void build_lm_head_cutlass_(cudaStream_t stream);

    // NVFP4 view of the LM head for the MTP draft chain's M=1 logits GEMV.
    // Fills `out` from the secondary decode cache (wcache_.nvfp4) or the
    // native-NVFP4 registry tier — the same sources the decode-path LM head
    // dispatch uses. Returns false when the LM head is not NVFP4-served
    // (cache disabled, FP8 LM head, or quantized source without a cache
    // entry); callers then keep the FP16 GEMV. The returned pointers borrow
    // executor-owned storage — do not free.
    bool lm_head_nvfp4_view(NvFP4QuantResult& out) const;

    // Set KV layer mapping (must be called before forward pass for hybrid models)
    void set_kv_layer_map(std::vector<int> map) {
        kv_layer_map_ = std::move(map);
        // Count KV layers and initialize per-layer FP8 scale vectors
        int n_kv = 0;
        for (int idx : kv_layer_map_) {
            if (idx >= 0)
                n_kv = std::max(n_kv, idx + 1);
        }
        kv_scales_.assign(n_kv, 1.0f);
        kv_calibrated_.assign(n_kv, false);
    }

    // Drop both the calibrated_ flag and the per-layer scale so the next
    // prefill recalibrates from a clean slate. Call this after warmup —
    // synthetic BOS tokens produce unrepresentative K/V absmax statistics
    // (Llama: too-small absmax, scale locked too tight, real data
    // overflowed FP8_MAX → degenerate output; Gemma-4: too-large absmax
    // from extreme output_norm outliers, scale locked too wide, real
    // data quantized to too-coarse FP8 grid → "Federer" garbage).
    // Resetting the scale value (not just the flag) avoids both failure
    // modes. High-water-mark within a single generation still applies
    // via the std::max in executor_kv_write.cu.
    void reset_kv_calibration() {
        std::fill(kv_scales_.begin(), kv_scales_.end(), 1.0f);
        std::fill(kv_calibrated_.begin(), kv_calibrated_.end(), false);
    }

    // Set layer offload manager (optional, for weight offloading)
    void set_offload_manager(LayerOffloadManager* mgr) { offload_mgr_ = mgr; }

    // Resize workspace for a different max token count (Phase 4: decode-mode optimization).
    // Uses cudaFreeAsync/cudaMallocAsync for near-instant resize via CUDA memory pool.
    [[nodiscard]] bool resize_workspace(int new_max_tokens, cudaStream_t stream) {
        return ws_.resize_workspace(new_max_tokens, stream);
    }

    // Dual workspace for concurrent prefill/decode overlap.
    // allocate_decode_workspace: creates a second workspace for decode (up to max_batch tokens).
    // use_workspace(0) = prefill (default), use_workspace(1) = decode.
    bool allocate_decode_workspace(cudaStream_t stream, int max_batch = 1) {
        return ws_.allocate_decode_workspace(stream, max_batch);
    }
    void use_workspace(int slot) { ws_.use_workspace(slot); }  // 0=prefill, 1=decode
    bool has_decode_workspace() const { return ws_.has_decode_workspace(); }
    int active_workspace() const { return ws_.active(); }
    int max_tokens() const { return max_tokens_; }

    // Capacity of the [n_heads, attn_seq, attn_seq] FP16 attn-scores workspace.
    // Engine's chunked-prefill path must clamp chunk_len so n × ctx_len ≤ cap².
    // Returns 0 if the buffer wasn't allocated (VRAM-constrained / WMMA fallback).
    int attn_scores_cap() const {
        return attn_scores_buf_ ? static_cast<int>(attn_scores_.shape[1]) : 0;
    }

    // Attention shapes are uniform when the per-layer shape arrays are absent
    // or carry a single distinct nonzero value (GDN/Mamba2 hybrids fill zeros
    // for non-attention layers). Uniform models can be served by the O(n)
    // FA2/FMHA chunked-prefill family; only truly heterogeneous shapes
    // (Gemma-4 dual head_dim 256/512) require the rectangular cuBLAS path.
    bool attn_shapes_uniform() const;

    // Largest prefill chunk starting at `offset` that the chunked-attention
    // dispatch can serve without overflowing the cuBLAS S-matrix (constraint:
    // n × (offset + n) ≤ s_cap² and n ≤ s_cap), floored to a kv_bs multiple.
    // Chunks served by the O(n) FA2/FMHA paths need no S-matrix and return
    // `desired` unchanged. Returns 0 when even a kv_bs-sized chunk cannot be
    // served (caller must reject the request instead of letting the kernel
    // capacity guard abort).
    int max_safe_prefill_chunk(int offset, int desired, int kv_bs) const;

    // Graph-captured verify chunk (#847). The chunked continuation forward is
    // replayable (InferenceState::ctx_capacity mode) only when the FP16-QK FA2
    // kernel serves EVERY attention layer with device-read lengths: uniform
    // hd=128 shapes, no learned sinks, no MLA, no LongRoPE (host branch on
    // max_context_len selects the freq table), fa2_fp16qk not disabled.
    bool chunk_capture_supported() const;
    // Persistent K/V scratch for the replayable chunked continuation
    // ([ctx_capacity, nkv, hd] FP16 each) — replaces the per-layer
    // cudaMallocAsync whose size would bake the growing ctx_len into the
    // graph. Idempotent; returns false on allocation failure.
    [[nodiscard]] bool ensure_chunk_capture_scratch(int ctx_capacity);
    // Bumped whenever the shared forward workspace is reallocated. A captured
    // verify graph holds raw pointers into it — the engine invalidates its
    // graphs when this changes.
    uint64_t workspace_generation() const { return ws_.generation(); }

    // Get a view of the logits buffer for n tokens (for CUDA graph replay,
    // where forward_logits isn't called but the graph writes to this buffer).
    Tensor get_logits_view(int n) const { return view_tokens(logits_, n); }

    // QJL projection accessor removed (TurboQuant retired Phase 5, 2026-05-17).

    // Release the MoE batch dequant buffer when expert weights are on host.
    // Call after weight upload if experts didn't fit on GPU.
    void release_moe_batch_buf();

    // Pre-allocated device buffer for sampling output (stable address for CUDA graph).
    int32_t* d_sample_result() const { return d_sample_result_; }

    // Pinned host buffer for logprobs extraction.
    float* h_logits_pinned() const { return h_logits_pinned_; }

    // Ensure pinned logits buffer is allocated for the given number of floats.
    // For single sequence: pass vocab_size. For batched logprobs: pass vocab_size * n_sequences.
    void ensure_logits_pinned(int total_floats);

    // Configure StreamingLLM smart KV cache: keeps the first `n_sinks` tokens
    // and the last `window` tokens of every attended sequence, dropping the
    // rest. Set n_sinks=0 to disable. Currently honoured only by the FP16 GQA
    // decode kernel; quantized variants ignore this and fall back to plain
    // sliding-window attention.
    void set_streaming_kv(int n_sinks, int window) {
        streaming_n_sinks_ = (n_sinks > 0) ? n_sinks : 0;
        streaming_window_ = (window > 0) ? window : 0;
    }
    int streaming_n_sinks() const { return streaming_n_sinks_; }
    int streaming_window() const { return streaming_window_; }

    // LoRA runtime delta (issue #522): activation-path low-rank deltas, no
    // weight patching — works with every quant tier. nullptr = base model.
    // The caller (Engine) must invalidate decode graphs around swaps: the
    // captured graph holds the adapter's kernel launches/pointers.
    void set_lora(const LoraAdapter* adapter);
    const LoraAdapter* lora() const { return lora_; }

    // Access the hidden state buffer after forward_logits().
    // Returns [max_tokens, d_model] FP16 on device. Use view_tokens() to get [n, d_model].
    const Tensor& hidden_state() const { return hidden_; }

    // Public view_tokens wrapper for external callers.
    Tensor view_hidden(int n_tokens) const { return view_tokens(hidden_, n_tokens); }

    // Phase 5 Track D (follow-up): per-Engine RuntimeConfig (the former
    // RuntimeConfig::current() singleton is gone). Engine wires this via
    // set_runtime_config() during init; the contract is now "set before
    // first access". Tests that build a bare GraphExecutor without an
    // owning Engine must wire a RuntimeConfig themselves (see
    // tests/test_helpers.h for a default loader).
    void set_runtime_config(const RuntimeConfig& cfg) noexcept { runtime_config_ = &cfg; }
    const RuntimeConfig& runtime_config() const noexcept {
        // CRITICAL: set_runtime_config() must be called before any forward.
        // Hard-failing here would crash unit tests; cold default is acceptable.
        static const RuntimeConfig kDefault;
        return runtime_config_ ? *runtime_config_ : kDefault;
    }

private:
    // The init-time weight-quantization pipeline (the 23 pre_dequant_*/
    // nvfp4_decode_* methods + the build-only StoragePlan) was extracted to
    // QuantPipeline (exec/quant_pipeline.h). GraphExecutor owns one
    // (quant_pipeline_, below) and delegates pre_dequant_weights() to it.

    // StreamingLLM (sinks + window). 0 = disabled.
    int streaming_n_sinks_ = 0;
    int streaming_window_ = 0;

    class VRAMAllocator* vram_alloc_ = nullptr;
    const Model* model_ = nullptr;
    QType compute_dtype_ = QType::F16;
    float norm_w_off_ = 0.0f;          // Gemma: 1.0 (norms use w+1 instead of w)
    void* v_norm_ones_buf_ = nullptr;  // Gemma 4: ones buffer for V-norm (no learned weight)
    bool initialized_ = false;
    int max_tokens_ = 0;

    // Shared eval-side LM-head driver: applies the tier-aware LM head to
    // hidden_[0..n_rows) in batches of max_logit_tokens_ and calls
    // consume(logits_view, row0, csz) after each batch's logits (softcap
    // already applied) land in logits_. Used by perplexity_nll_partial and
    // greedy_argmax_all.
    void for_each_lm_head_batch_(int n_rows, cudaStream_t stream,
                                 const std::function<void(const Tensor&, int, int)>& consume);

    // Spec-decode verify argmax partials (lazy, grows only; freed in
    // free_buffers).
    void* verify_argmax_scratch_ = nullptr;
    size_t verify_argmax_scratch_sz_ = 0;
    // Spec-decode verify penalty counts: [vocab] int32 occurrence counts of
    // the shared history (lazy; freed in free_buffers).
    int32_t* verify_pen_counts_ = nullptr;
    int verify_pen_counts_cap_ = 0;

    // LoRA (issue #522)
    const LoraAdapter* lora_ = nullptr;
    void* lora_scratch_ = nullptr;  // fp32[max_rank] + fp16[max_tokens*max_rank]
    size_t lora_scratch_sz_ = 0;
    void lora_delta_(const LoraWeights& w, const void* x, void* y, int n, cudaStream_t stream);
    int max_logit_tokens_ = 0;     // max tokens needing LM head projection (= max(max_batch_size, 8))
    int cur_n_tokens_ = 0;         // set by forward_logits for use by run_ffn
    int cur_decode_step_ = 0;      // set by forward_logits for debug dump tagging
    bool cur_force_fp16_ = false;  // set by forward_logits, bypasses FP8 GEMM paths
    bool cur_per_row_lm_ = false;  // set by forward_logits, per-row Q8_1 LM head

    // Programmatic Dependent Launch: when true, custom kernels have the PDL
    // attribute set so the GPU can overlap tail of one kernel with head of next.
    bool use_pdl_ = false;

    // --- Scratch arena (shared/persistent/decode workspace) ---
    // Owns the workspace buffers + sizes + the decode/prefill swap state; the
    // moved methods write the activation/phase tensors below through pointers
    // set in ws_.init() (called from GraphExecutor::init). See exec/workspace.h.
    Workspace ws_;

    // Persistent activation tensors (views into the persistent workspace)
    Tensor hidden_;    // [max_tokens, d_model] FP16
    Tensor residual_;  // [max_tokens, d_model] FP16
    Tensor norm_out_;  // [max_tokens, d_model] FP16
    Tensor logits_;    // [max_logit_tokens, vocab_size]

    // FP32 residual accumulator for post-norm architectures (Gemma-3).
    // Prevents FP16 overflow in the residual stream over many layers.
    // The FP32 tensor is the "true" hidden state; the FP16 hidden_ is only
    // used as input to RMSNorm (which is scale-invariant, so clamping is safe).
    // nullptr for pre-norm models (LLaMA, Qwen, etc.).
    void* fp32_accum_buf_ = nullptr;
    Tensor fp32_hidden_;  // [max_tokens, d_model] FP32 — true hidden state

    // The shared/persistent workspace buffers + per-phase sizes live in ws_.
    // The phase TENSORS below are views carved by GraphExecutor's
    // configure_*_workspace methods (which slice ws_.shared()); the hot path
    // reads them as members.

    // Attention phase tensors (views into the shared workspace, set by configure_attn_workspace)
    Tensor q_;         // [max_tokens, n_heads * head_dim]
    Tensor k_;         // [max_tokens, n_kv_heads * head_dim]
    Tensor v_;         // [max_tokens, n_kv_heads * head_dim]
    Tensor attn_out_;  // [max_tokens, n_heads * head_dim]
    Tensor proj_out_;  // [max_tokens, d_model]

    // cuBLAS attention S-matrix workspace (separately allocated, not part of shared workspace).
    // [n_heads, max_tokens, max_tokens] FP16 — used only during prefill.
    void* attn_scores_buf_ = nullptr;
    size_t attn_scores_buf_size_ = 0;
    Tensor attn_scores_;  // 3D tensor view into attn_scores_buf_

    // Persistent K/V scratch for the graph-captured verify chunk (#847),
    // sized by ensure_chunk_capture_scratch. chunk_capture_ctx_ is the
    // ctx_capacity the buffers cover (0 = unallocated).
    half* chunk_capture_k_ = nullptr;
    half* chunk_capture_v_ = nullptr;
    int chunk_capture_ctx_ = 0;

    // Dense FFN phase tensors (views into shared_workspace_, set by configure_ffn_workspace)
    Tensor gate_out_;    // [max_tokens, d_ff]
    Tensor up_out_;      // [max_tokens, d_ff]
    Tensor swiglu_out_;  // [max_tokens, d_ff]
    Tensor ffn_out_;     // [max_tokens, d_model]

    // MoE workspace (phase tensors + separately allocated buffers)
    MoEWorkspace moe_;

    // SSM phase tensors (views into shared_workspace_, set by configure_ssm_workspace)
    Tensor ssm_proj_buf_;  // [max_tokens, ssm_in_dim] for ssm_in projection
    Tensor ssm_xBC_buf_;   // [max_tokens, conv_channels] for conv output
    Tensor ssm_y_buf_;     // [max_tokens, inner_size] for scan output
    Tensor ssm_z_buf_;     // [max_tokens, inner_size] for gate
    Tensor ssm_out_buf_;   // [max_tokens, d_model] for ssm_out projection
    Tensor ssm_dt_buf_;    // [max_tokens, n_heads] for dt after split
    Tensor gdn_fused_proj_buf_;  // [max_tokens, conv_channels+inner+2*n_heads] FP16 — output
                                 // of the fused GDN input GEMV when ly.gdn_input_packed is
                                 // built; sized only for has_gdn_ models.

    // --- Separately allocated buffers (not part of unified workspace) ---

    // LRU cache for host-resident expert weights on GPU.
    // Keeps recently-used experts in VRAM to avoid repeated H2D copies.
    ExpertLRUCache expert_cache_;

    // Pre-allocated dequant scratch for the gemm_nvfp4 fallback (M>1 only).
    // Set up by allocate_nvfp4_dequant_workspace() and registered with the
    // free function via set_nvfp4_dequant_workspace(). Allows the fallback
    // path to run inside CUDA stream capture without crashing on cudaMalloc.
    void* nvfp4_dequant_ws_buf_ = nullptr;
    size_t nvfp4_dequant_ws_size_ = 0;

    // MLA (DeepSeek) persistent QKV scratch — pre-allocated once (sized for
    // max_tokens) so the materialized two-step KV projection never calls
    // cudaMallocAsync inside the CUDA-graph-captured decode region (capture
    // rejects stream-ordered alloc/free → silent eager fallback + degeneration).
    void* mla_kv_a_buf_ = nullptr;    // [max_tokens, kv_lora_rank + qk_rope_head_dim]
    void* mla_latent_buf_ = nullptr;  // [max_tokens, kv_lora_rank]
    void* mla_k_rope_buf_ = nullptr;  // [max_tokens, qk_rope_head_dim]
    void* mla_kv_b_buf_ = nullptr;    // [max_tokens, n_heads*(qk_nope_head_dim+v_head_dim)]

    // MLA absorbed-decode latent KV cache (Phase 3, opt-in attention.mla_absorb).
    // Per-layer slice = [max_seq, kv_lora_rank + qk_rope_head_dim] FP16; cols
    // [0:kv_lora_rank] = RMSNorm'd latent, [kv_lora_rank:] = post-RoPE decoupled
    // key. Allocated only when mla_absorb is set, the model is_mla(), and every
    // layer's kv_b_proj is FP16. Single-sequence only.
    void* mla_absorb_cache_ = nullptr;        // [n_layers, max_seq, kv_lora+rope]
    float* mla_absorb_scores_ = nullptr;      // [n_heads, max_seq] decode scratch
    size_t mla_absorb_layer_stride_ = 0;      // halfs per layer = max_seq*(kv_lora+rope)
    int mla_absorb_max_seq_ = 0;

    // Set when allocate_nvfp4_dequant_workspace() could NOT pre-allocate the
    // M>1 dequant scratch (largest NVFP4 weight exceeds the cap, or alloc
    // failed). The fallback then lazy-cudaMallocs, which is illegal inside
    // CUDA graph capture (cublasLt status 14 → cascading capture failure).
    // The scheduler reads this to skip prefill-graph capture (run eager).
    bool nvfp4_dequant_uncapturable_ = false;

public:
    bool nvfp4_dequant_uncapturable() const { return nvfp4_dequant_uncapturable_; }

private:
    // Init-time weight-quantization pipeline (D2 extraction). Owns the build-only
    // StoragePlan; fills the long-lived caches below by reference in build().
    QuantPipeline quant_pipeline_;

    // Weight caches (FP16, FP8, NVFP4, CUTLASS NVFP4/MXFP4, fused KV/gate+up)
    WeightCaches wcache_;

    // Mode flags mirrored from wcache_ for PlanHints (hints_ is the Phase 5 source of truth).
    PlanHints hints_;

    // The build-only StoragePlan (storage_plan_), plan_tier_of(), and
    // apply_arch_rules_() moved into QuantPipeline (exec/quant_pipeline.h) —
    // they have no hot-path reader.

    // WeightRegistry: parallel handle store (Phase 2+ shim, populated alongside wcache_)
    WeightRegistry registry_;

    // Quantization scratch buffers (FP8 act, CUTLASS act, dp4a, dequant, split-K)
    QuantScratch qscratch_;

    // CUTLASS NVFP4 LM head for batched-decode (n>1) tensor-core GEMM. Borrows
    // the FP4 data from the NVFP4 decode cache; owns only the SfAtom scales.
    // Built by build_lm_head_cutlass_() when serving; freed in the destructor.
    CutlassNvFP4Weight lm_head_cutlass_{};
    bool lm_head_cutlass_ready_ = false;

    // Pre-allocated sampling result buffers (avoids cudaMalloc/cudaFree per token).
    int32_t* d_sample_result_ = nullptr;  // device buffer for argmax/sample kernel output
    int32_t* h_sample_pinned_ = nullptr;  // pinned host buffer for async D2H sample result

    // Pinned host buffer for logprobs extraction (D2H copy of logits)
    float* h_logits_pinned_ = nullptr;  // [vocab_size] pinned host memory
    int h_logits_pinned_size_ = 0;      // vocab_size used for allocation

    // --- Layer index mappings ---

    // Mapping from global layer index to SSM layer index (for SSMState access)
    std::vector<int> ssm_layer_map_;  // ssm_layer_map_[global_idx] = ssm_idx, or -1

    // Mapping from global layer index to GDN layer index (for GDNState access)
    std::vector<int> gdn_layer_map_;  // gdn_layer_map_[global_idx] = gdn_idx, or -1

    // Mapping from global layer index to KV cache layer index (for attention layers only)
    std::vector<int> kv_layer_map_;  // kv_layer_map_[global_idx] = kv_idx, or -1

    // Per-KV-layer FP8 scales for online calibration.
    // Scale = absmax / 448.0; used as inv_scale = 1/scale for write, scale for read.
    std::vector<float> kv_scales_;     // [n_kv_layers] per-layer FP8 scale
    std::vector<bool> kv_calibrated_;  // [n_kv_layers] whether scale has been calibrated

    // YaRN correction dimension boundaries [2], precomputed at init.
    // yarn_corr_dims_[0] = start (full interpolation below), yarn_corr_dims_[1] = end (full extrapolation
    // above)
    float yarn_corr_dims_[2] = {0.0f, 0.0f};

    // LongRoPE pre-computed inverse frequencies (device memory)
    float* longrope_short_freqs_ = nullptr;  // [rope_pairs] device
    float* longrope_long_freqs_ = nullptr;   // [rope_pairs] device
    int longrope_orig_max_pos_ = 0;
    int longrope_n_pairs_ = 0;

    // --- Model feature flags (set during init for workspace computation) ---
    bool has_moe_ = false;
    bool has_ssm_ = false;
    bool has_gdn_ = false;
    bool has_dense_ffn_ = false;

    // Max expert FFN hidden dim from actual packed tensor shapes (may differ from cfg.expert_d_ff)
    int max_expert_eff_ = 0;

    // The dual prefill/decode workspace swap state (decode_workspace_,
    // decode_shared_workspace_, the sizes, decode_max_batch_, active_workspace_,
    // SavedWorkspace saved_prefill_ws_) moved into ws_ (exec/workspace.h).

    // --- Layer offload manager (non-owning, set by engine) ---
    LayerOffloadManager* offload_mgr_ = nullptr;

    // --- Per-Engine RuntimeConfig (Phase 5 Track D, non-owning) ---
    // Engine wires this via set_runtime_config() during Engine::init.
    // Replaces RuntimeConfig::current() inside GraphExecutor::* methods.
    const RuntimeConfig* runtime_config_ = nullptr;

    // --- Allocation and configuration methods ---
    // The shared/persistent/decode scratch arena (allocate_*_workspace,
    // compute_shared_sizes, workspace_estimate, resize_workspace,
    // allocate_decode_workspace, use_workspace) moved into Workspace
    // (exec/workspace.h); GraphExecutor owns ws_ and delegates.
    //
    // The four per-phase carvers stay here: they write GraphExecutor's
    // activation-tensor members (q_/k_/v_/.../gdn_fused_proj_buf_) by slicing
    // the shared buffer (read via ws_.shared()), so they belong on GraphExecutor.
    void configure_attn_workspace(int max_tokens);
    void configure_ffn_workspace(int max_tokens);
    void configure_moe_workspace(int max_tokens);
    void configure_ssm_workspace(int max_tokens);

    void allocate_auxiliary_buffers(
        bool skip_batch_dequant = false);  // dequant scratch, MoE staging, routing buffers
    void free_buffers();

    // Per-layer helpers
    void run_attention(int layer, const InferenceState& state, cudaStream_t stream);
    void run_ffn(int layer, cudaStream_t stream);
    void run_moe_ffn(int layer, cudaStream_t stream);

    // 5.1.3.d transitional: route M>1 prefill through WeightHandle dispatch
    // (tier-aware, no raw-data deref), M=1 decode through legacy dispatch
    // (dp4a on original quant is fastest). Caller passes both the TensorID
    void gemm_via_handle_(TensorID id, const Tensor& input,
                          Tensor& output, const GemmContext& ctx);
    // True when an M>1 dispatch of `id` is guaranteed to take the CUTLASS
    // NVFP4 prefill block in gemm_via_handle_ (which quantizes the input
    // into the shared activation scratch). Gate for the act-quant-hint
    // dedupe at the QKV / gate-up call sites — must be CONSERVATIVE: a
    // false negative just re-quantizes; a false positive would skip with a
    // stale scratch.
    bool prefill_routes_cutlass_nvfp4_(TensorID id) const;
    // MoE forward-pass phase helpers. The per-call locals live in the
    // MoeFfnContext struct declared just above the GraphExecutor class.
    void moe_ffn_phase1_setup_(int layer, cudaStream_t stream);
    void moe_ffn_phase2_state_and_norm_(int layer, cudaStream_t stream, MoeFfnContext& ctx);
    void moe_ffn_phase3_route_(int layer, cudaStream_t stream, MoeFfnContext& ctx);
    void moe_ffn_phase7_scatter_(int layer, cudaStream_t stream, MoeFfnContext& ctx);
    void moe_ffn_phase8_post_(int layer, cudaStream_t stream, MoeFfnContext& ctx);
    // NVFP4→FP16 batch dequant fallback (when CUTLASS 3.x grouped-NVFP4
    // can't fire). Predicate is checked internally; returns true if the
    // path ran.
    bool try_run_moe_nvfp4_dequant_batch_prefill_(int layer, cudaStream_t stream, MoeFfnContext& ctx);
    // Legacy MoE prefill path (D2H-sync + serial / batched-dequant /
    // pre-cached-FP16 dispatch). Unconditional — caller selects when no
    // other path matched.
    void run_moe_legacy_fallback_(int layer, cudaStream_t stream, MoeFfnContext& ctx);
    // CUTLASS 3.x NVFP4 BlockScaled grouped GEMM MoE prefill (device-args
    // / smallM / legacy host-args sub-variants). Predicate is checked
    // internally; returns true if the path ran.
    bool try_run_moe_cutlass3x_nvfp4_prefill_(int layer, cudaStream_t stream, MoeFfnContext& ctx);
    // Cheap precondition mirror for try_run_moe_cutlass3x_nvfp4_prefill_'s
    // device-args fast path. Read upstream of moe_gather to skip the gather
    // when the path is guaranteed to fire (and thereby own this MoE layer
    // exclusively — it reads ctx.no via sorted_token_ids and doesn't need
    // the gathered intermediate). If the device-args path's run-time gate
    // turns out false anyway, the legacy fallback gathers lazily — so a
    // mismatch here costs at most one wasted gather, never wrong output.
    bool moe_cutlass3x_will_use_device_args_(int layer, const MoeFfnContext& ctx) const;
    // Optional shared expert (parallel dense FFN) — called from run_moe_ffn
    // after routed experts have written into h. Reads `no` (post-norm) and
    // adds its result back into `h` via elementwise_add. No-op when
    // ly.w_up_shared is null or the runtime opt-out is set.
    void run_shared_expert_ffn(int layer, cudaStream_t stream, int n, int d,
                               float eps, const Tensor& no, Tensor& h);
    // MoE decode fast-path (n=1, device-resident packed experts):
    // dispatches all top_k experts in a single kernel per projection. NVFP4
    // and dp4a/FP16 sub-paths handled internally. Sets `residual_fused`=true
    // when the weighted sum fused the residual add (no shared expert path
    // active). Caller invokes only when decode_fast eligibility predicate
    // returned true.
    void run_moe_decode_fast(int layer, cudaStream_t stream, int n, int d, int eff,
                             int top_k, const MoeRoutingResult& routing,
                             const Tensor& no, Tensor& h, const Tensor& r,
                             bool moe_use_fp32_residual, bool moe_fused_norm_q8,
                             bool will_skip_residual_copy, bool& residual_fused);
    // Compute MoE routing: gate logits (FP32 router fast-path for Gemma-4
    // already done by caller — signaled via `fp32_gate_logits_ready`) + topk
    // gating + per-expert weight scaling (Nemotron, Gemma-4). Caller passes
    // pre-normalized `router_in` if !fp32_gate_logits_ready.
    void compute_moe_routing(int layer, cudaStream_t stream, int n, int d, int ne,
                             int top_k, const Tensor& router_in,
                             bool fp32_gate_logits_ready, bool will_decode_fast,
                             const void* router_bias_ptr, bool use_sigmoid,
                             bool norm_weights, MoeRoutingResult& routing);
    // Fused Q6_K prefill MoE path: reads Q6_K weights directly (no FP16
    // dequant scratch), TC variant uses gather-free sorted_token_ids
    // indirection, scalar variant materializes the gathered buffer.
    // Fills moe_.expert_{gate,up,swiglu,down} for downstream scatter.
    // Returns true when path was taken (caller skips general path branches).
    bool try_run_moe_q6k_prefill(int layer, cudaStream_t stream, int n, int d, int eff,
                                 int ne, int expanded, bool non_gated_experts, QType up_qtype,
                                 const MoeRoutingResult& routing, const Tensor& no);
    // Fused Q4_K prefill: reads Q4_K weights directly, FP16 activations from
    // L1/L2 cache. Same interface as Q6_K but with Q4_K dequant logic.
    bool try_run_moe_q4k_prefill(int layer, cudaStream_t stream, int n, int d, int eff,
                                 int ne, int expanded, bool non_gated_experts, QType up_qtype,
                                 const MoeRoutingResult& routing, const Tensor& no);
    // Gemma-4 ggml MMVQ per-token prefill: processes tokens individually via
    // ggml Q4_K×Q8_1 dp4a kernels with FP32 norm output for full-precision
    // routing. Writes directly to `h` (per-token weighted sum + residual).
    // Returns true when path was taken; caller jumps to moe_after_experts.
    bool try_run_moe_gemma4_ggml_prefill(int layer, cudaStream_t stream, int n, int d, int eff,
                                         int top_k, QType up_qtype, float eps,
                                         const MoeRoutingResult& routing, const Tensor& no,
                                         const Tensor& norm_w, Tensor& h, const Tensor& r,
                                         bool moe_use_fp32_residual, bool& residual_fused);
    // FP16 batch dequant + cublasGemmGroupedBatchedEx prefill: dequants all
    // experts to FP16 in one shot, runs a single grouped GEMM per projection.
    // One D2H sync per layer for offsets (unavoidable for grouped GEMM API).
    // Falls through to scatter (caller's responsibility); returns true when
    // path was taken.
    bool try_run_moe_fp16_batch_prefill(int layer, cudaStream_t stream, int n, int d, int eff,
                                        int ne, int expanded, bool non_gated_experts,
                                        QType up_qtype, const MoeRoutingResult& routing,
                                        bool fp32_down_active, void*& fp32_down_buf);
    // FP8 batch prefill: Q6_K → FP8 dequant, per-expert FP16→FP8 quantize,
    // cuBLAS FP8 grouped GEMM → FP16. Falls back to FP16 batch when scales
    // unavailable. Used when FP16 batch buffer can't fit but FP8 can.
    bool try_run_moe_fp8_batch_prefill(int layer, cudaStream_t stream, int n, int d, int eff,
                                       int ne, int expanded, bool non_gated_experts,
                                       QType up_qtype, const MoeRoutingResult& routing);
    void run_ssm(int layer, const InferenceState& state, cudaStream_t stream);
    void run_gdn(int layer, const InferenceState& state, cudaStream_t stream);

    // Layer type detection (based on tensor presence)
    bool layer_has_attention(int layer) const;
    bool layer_has_ssm(int layer) const;
    bool layer_has_gdn(int layer) const;
    bool layer_has_moe(int layer) const;
    bool layer_has_dense_ffn(int layer) const;

    // Write computed K/V into KV cache blocks
    void write_kv_cache(int layer, const InferenceState& state, cudaStream_t stream);

    // Create a Tensor view of the first n_tokens rows of a max_tokens buffer.
    Tensor view_tokens(const Tensor& buf, int n_tokens) const;
};

}  // namespace imp
