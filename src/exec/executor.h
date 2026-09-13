#pragma once

#include "core/dispatch_policy.h"

#include "model/model.h"
#include "memory/host_pinned.h"
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
#include "compute/sampling.h"  // TopkRowArgs (row-batched sampler staging)
#include "exec/activation_calibrator.h"
#include "exec/expert_cache.h"
#include "exec/nvfp4_expert_offload.h"
#include "exec/inference_state.h"
#include "exec/moe_ffn_context.h"
#include "exec/weight_caches.h"
#include "exec/weight_handle.h"
#include "exec/moe_workspace.h"
#include "exec/quant_scratch.h"
#include "exec/quant_pipeline.h"
#include "exec/workspace.h"
#include "exec/storage_planner.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <span>
#include <unordered_map>
#include <utility>
#include <list>

namespace imp {

class LoraAdapter;
struct LoraWeights;

// Pipelined batched-decode chain advance (decode_pipeline_advance.cu): see
// decode_pipeline_advance.cu for the mechanism. Patch/pos arrays must be
// device-readable (mapped pinned memory). Declared here (host-safe) so
// host-only TUs (engine_scheduler.cpp) can launch it.
void decode_pipeline_advance(int n_rows, const int32_t* slot_tokens, size_t slot_stride_bytes,
                             int32_t* d_token_ids, int* d_positions, int* d_context_lens,
                             int* d_block_tables, int n_patches, const int* d_patch_offsets,
                             const int* d_patch_values, int32_t* d_hist_base, int hist_stride,
                             const int* d_hist_pos, cudaStream_t stream);

// Nvfp4DecodeContext moved to exec/quant_pipeline.h (build-only; consumed by
// the QuantPipeline phase-3 helpers).

// Imperative executor for the transformer forward pass. The Graph class is
// a DAG for visualization/debugging only; this executor hardcodes the
// standard forward pass for efficiency. No graph walking at runtime.
struct GemmContext;  // defined in gemm_context.h
struct VRAMBudget;   // defined in runtime/vram_budget.h; only pre_dequant_weights() names it
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

    // Estimated GPU memory needed by allocate_workspaces(), used by Engine for
    // the expert upload reserve. The S-matrix term is included only when the
    // allocator would actually build one (#943): FA2-served configs skip it, so
    // charging it here would hold phantom headroom out of the cache/KV planners.
    size_t workspace_estimate() const {
        return ws_.workspace_estimate(/*include_attn_scores=*/!fa2_serves_all_prefill());
    }

    // Run the full forward pass and return the sampled token ID.
    int32_t forward(const InferenceState& state, cudaStream_t stream = nullptr);

    // Run the forward pass but return raw logits instead of sampling.
    // logits_out will be a view into the internal logits buffer.
    void forward_logits(const InferenceState& state, Tensor& logits_out, cudaStream_t stream = nullptr);

    // Teacher-forced perplexity over tokens[0..n-1]. Call AFTER a SINGLE-CHUNK
    // prefill of the same tokens (uses the persistent hidden_). Bench/eval
    // only. For chunked prefill use Engine::begin/end_perplexity_capture,
    // which accumulates via perplexity_nll_partial after every chunk.
    double perplexity_nll(std::span<const int32_t> tokens, cudaStream_t stream = nullptr);

    // Per-chunk NLL accumulation: applies the tier-aware LM head to
    // hidden_[0..chunk_len-1] (absolute corpus positions chunk_start..+len-1),
    // writes -log p(next token) into d_nll[global_pos]. Enqueues on `stream`
    // only; caller reduces after a sync. Overwrites logits_: call only after
    // all reads of the chunk's logits are done. Optional d_match: greedy argmax == actual next token.
    void perplexity_nll_partial(const int32_t* d_tokens, int n_total, int chunk_start,
                                int chunk_len, double* d_nll, cudaStream_t stream,
                                int32_t* d_match = nullptr);

    // Embedding pooling (#1005): column-wise sum of hidden_[0..n_tokens) into
    // d_out[d_model] fp32. Runs after a chunk's forward while hidden_ still
    // holds it; engine accumulates chunk sums host-side for full-input pooling.
    void pool_hidden_sum(int n_tokens, float* d_out, cudaStream_t stream);

    // Greedy verify for n-gram speculative decoding: applies the tier-aware LM
    // head to hidden_[0..n_rows-1] (same logits as production greedy, incl.
    // softcap), writes argmax to d_out. Overwrites logits_.
    // Optional penalties: d_hist ends with the chunk's t0; d_draft holds chunk
    // tokens after t0, so row j also penalizes d_draft[0..j-1] (production parity).
    // d_topm/topm: writes each row's top-M post-penalty logit ids, harvested by
    // the Token-Recycling adjacency drafter.
    // allow_cutlass: batched-decode W4A4 LM head instead of the per-row GEMV.
    // d_banned_alt/h_row_alt: per-request think-stop mask for rows where
    // h_row_alt[row]!=0, in place of d_banned.
    // h_row_hist/h_row_pens: per-row penalty histories for a batched chunk's
    // mixed-request rows, applied before the ban and argmax. Exclusive with d_hist.
    void greedy_argmax_all(int n_rows, int32_t* d_out, cudaStream_t stream,
                           const int32_t* d_hist = nullptr, int n_hist = 0,
                           const int32_t* d_draft = nullptr, float rep_pen = 1.0f,
                           float freq_pen = 0.0f, float pres_pen = 0.0f,
                           int32_t* d_topm = nullptr, int topm = 0,
                           const int32_t* d_banned = nullptr, int n_banned = 0,
                           bool allow_cutlass = false, const int32_t* d_banned_alt = nullptr,
                           int n_banned_alt = 0, const uint8_t* h_row_alt = nullptr,
                           const int32_t* const* h_row_hist = nullptr, const int* h_row_hist_n = nullptr,
                           const float* h_row_pens = nullptr);

    // LM head + argmax over EXTERNAL, already-normed rows ([n_rows,d_model]
    // FP16, e.g. MTP head output): the batched-decode W4A4 LM head, no output
    // norm applied. False (nothing written) when that head isn't built
    // (max_batch_size==1) or on a GEMM failure.
    bool lm_head_rows_argmax(const void* d_rows, int n_rows, int32_t* d_out, cudaStream_t stream);

    // Materializes the LM-head projection of hidden_[0..n_rows) into
    // d_out[n_rows,vocab_size] fp32 (same re-projection as greedy_argmax_all,
    // but copying rows instead of reducing). Consumed by the constrained-
    // pipeline jump-ahead (#844). Call while the chunk-forward workspace is still active.
    void project_logits_all(int n_rows, float* d_out, cudaStream_t stream);

    // Sample tokens from pre-computed logits (for use after CUDA graph execution).
    std::vector<int32_t> sample_from_logits(const Tensor& logits, const InferenceState& state,
                                            cudaStream_t stream = nullptr);

    // Single-token sampling, avoids a vector alloc; use for single-sequence
    // decode. Enqueue-only per-row sampling into scratch slot slot_idx (no
    // readback/sync); returns false for sync-only modes (mirostat, logit_bias,
    // CUB top_k) without touching the row. Gather via collect_sampled_tokens.
    bool sample_single_from_logits_async(const Tensor& logits, const InferenceState& state, int slot_idx,
                                         cudaStream_t stream = nullptr);
    const int32_t* collect_sampled_tokens(int n_slots, cudaStream_t stream = nullptr);
    // One-launch append of this step's sampled tokens (strided slots, ACTIVE
    // parity half) into per-request device penalty histories: row i maps
    // slot i -> hist[slots[i]*cap+offs[i]]; offs[i]<0 skips. Enqueue after the
    // row samplers, before the parity flip.
    bool append_sampled_history(const PenaltyAppendArgs& args, int32_t* d_hist,
                                cudaStream_t stream = nullptr);

    // Parity-buffered sampling for pipelined batched decode: slot region,
    // pinned gather buffer, and top-k staging are allocated x2; parity selects
    // the active half so step N+1's sampler chains enqueue into p^1 while step
    // N's tokens (p) are still in flight to host. gather_sampled_tokens_async
    // replaces the stream sync with an event; wait_gathered_tokens waits only
    // that event. Non-pipelined callers run at parity 0.
    void set_sample_parity(int parity) { sample_parity_ = parity & 1; }
    bool gather_sampled_tokens_async(int n_slots, cudaStream_t stream = nullptr);
    const int32_t* wait_gathered_tokens(int parity);
    // Device pointer of slot 0 of the given parity set (the chain-advance
    // kernel reads sampled tokens strided by SAMPLE_SCRATCH_BYTES from here).
    const int32_t* sample_slot_base(int parity) const;
    bool sample_pipeline_ready() const {
        return d_sample_result_ && !h_sample_pinned_.empty() && !h_row_args_.empty() && d_row_args_ &&
               sample_gather_evt_[0] && sample_gather_evt_[1];
    }
    int32_t sample_single_from_logits(const Tensor& logits, const InferenceState& state,
                                      cudaStream_t stream = nullptr);

    // Async decode: forward reads the token from device memory (d_token_id),
    // samples, writes back to d_token_id. No host-device sync; returns
    // immediately. h_mapped is pinned memory for host polling of the token.
    void forward_decode_async(const InferenceState& state, int32_t* d_token_id, int32_t* h_mapped,
                              cudaStream_t stream = nullptr);

    // Constrained async sampling (pipelined constrained decode): applies
    // device-side banned-token + active json/schema mask, samples on device.
    // Writes to d_result (must be SAMPLE_SCRATCH_BYTES) and async-copies to
    // h_pinned. No host-device sync; order via an event.
    void masked_sample_async(const InferenceState& state, const Tensor& logits, int32_t* d_result,
                             int32_t* h_pinned, cudaStream_t stream);

    // Set centralized VRAM allocator for budget-tracked allocations.
    // Must be called before allocate_workspaces() / pre_dequant_weights().
    void set_vram_allocator(class VRAMAllocator* alloc) { vram_alloc_ = alloc; }

    // Pre-dequantize quantized weights to FP16 on GPU for fast prefill GEMM.
    // Must be called AFTER model weights are uploaded to GPU.
    // budget: VRAM budget with per-phase caps computed by Engine::plan_vram_budget().
    void pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget);

    // Allocates the gemm_nvfp4 dequant workspace sized to the max NVFP4 weight.
    // Must run AFTER pre_dequant_weights(). Skips if no NVFP4 weights exist or
    // the largest exceeds the 512 MiB sanity cap; the fallback then lazy-cudaMallocs on non-captured streams.
    bool allocate_nvfp4_dequant_workspace();

    // Builds a CUTLASS NVFP4 LM-head weight for batched decode (n>1): one
    // tensor-core GEMM (weight read once) instead of per-row/batched-M GEMV
    // (read ceil(M/4)x). Only the SfAtom scale buffer is allocated; FP4 data
    // borrows the NVFP4 decode cache. No-op unless serving and LM head is NVFP4. Must run AFTER
    // pre_dequant_weights().
    void build_lm_head_cutlass_(cudaStream_t stream);

    // Split-K workspace for gemm_nvfp4_smallm (batched-decode small-M GEMMs).
    // Sized lazily on the first eager use, stable across graph capture.
    void* smallm_ws_ = nullptr;
    size_t smallm_ws_bytes_ = 0;
    void* smallm_xq_ = nullptr;   // A4 activation quantize scratch [32, Kmax]
    size_t smallm_xq_bytes_ = 0;
    bool smallm_arena_ = false;  // both scratches are T2 arena slabs: never cudaFree them
    // What smallm_xq_ currently holds (source pointer + shape of the last
    // activation quantize). Lets a dispatch with a matching hint skip
    // re-quantizing when two GEMMs share one normed input (gate/up, q/k/v, GDN
    // in/z). Self-invalidates on every fresh quantize.
    const void* smallm_xq_src_ = nullptr;
    int smallm_xq_src_m_ = 0;
    int smallm_xq_src_k_ = 0;
    // True when the scratch was filled by a PRODUCER fusion (fused
    // rmsnorm/swiglu + quantize), not the dispatch's own quantize: the
    // small-M block then accepts a matching tag without an act-quant hint,
    // since the producer updated the tag on the same write. Never read by the CUTLASS prefill consumer.
    bool smallm_xq_from_producer_ = false;

    // Grow the small-M xq scratch (no-op while capturing; resize
    // invalidates the shared-activation tag).
    void ensure_smallm_xq_(size_t xq_need, cudaStream_t stream);
    // Producer-fusion gate + scratch handout: returns the xq packed/scales
    // pointers when consumer_id's weight will take the small-M NVFP4 route for
    // [M,K] F16 activations and the scratch fits; nullptr otherwise (caller
    // runs unfused kernels). Never allocates while `stream` is capturing.
    uint8_t* smallm_producer_xq_(TensorID consumer_id, int M, int K, cudaStream_t stream,
                                 uint8_t** scales_out);
    // Tag the scratch as holding quantize(out[0..M,0..K)) written by a fused
    // producer kernel.
    void smallm_producer_tag_(const void* out_data, int M, int K);
    // The plain-NVFP4 weight the small-M GEMM reads for `h`: the native
    // CUTLASS_NVFP4 source, or on decode rows (#1897) the NVFP4 decode overlay
    // of a dequantable GGUF source. False when neither exists; prompt rows on
    // a GGUF source keep the full-precision dequant route. Single gate shared
    // by the dispatch block, sibling-pair dispatch and producer fusion.
    bool smallm_weight_(const WeightHandle& h, NvFP4QuantResult& out) const;
    // Grow the small-M GEMM workspace to `need` bytes; a no-op while
    // `stream` is capturing (twin of ensure_smallm_xq_).
    void ensure_smallm_ws_(size_t need, cudaStream_t stream);
    // Fused rmsnorm+quantize when the consumer takes the small-M route; falls
    // back to plain rmsnorm() internally. `consumer_id` is the FIRST GEMM
    // reading `no` (q/gate/GDN in); further readers skip via the act-quant hint.
    void rmsnorm_for_smallm_(const Tensor& h, const Tensor& w, Tensor& no, TensorID consumer_id,
                             int n, float eps, cudaStream_t stream, float weight_offset);
    // Fused swiglu+quantize when the down projection takes the small-M
    // route; falls back to plain swiglu() internally.
    void swiglu_for_smallm_(const Tensor& go, const Tensor& uo, Tensor& so, TensorID consumer_id,
                            int n, cudaStream_t stream);
    // NVFP4 view of the LM head for the MTP draft chain's M=1 logits GEMV:
    // fills `out` from the secondary decode cache or the native-NVFP4 registry
    // tier (same sources the decode-path LM head uses). False when the LM head
    // isn't NVFP4-served; callers then keep the FP16 GEMV. Returned pointers borrow executor storage, do not
    // free.
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

    // Drops calibrated_ AND the per-layer scale value (not just the flag) after
    // warmup: synthetic BOS tokens give unrepresentative K/V absmax (too-tight
    // scale overflows to FP8_MAX on real data, or too-wide scale under-uses
    // the FP8 grid). High-water-mark within one generation still applies via
    // std::max in executor_kv_write.cu.
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
    // 0=prefill, 1=decode. Also swaps the collision-prone quant-scratch family
    // (fp8 activation + q8_1) onto per-slot copies when the decode set exists
    // (required for prefill/decode overlap); no-op otherwise.
    void use_workspace(int slot);
    bool has_decode_workspace() const { return ws_.has_decode_workspace(); }
    int decode_max_batch() const { return ws_.decode_max_batch(); }
    int active_workspace() const { return ws_.active(); }
    int max_tokens() const { return max_tokens_; }

    // Prefill/decode overlap support. allocate_decode_qscratch: per-slot
    // copies of quant scratches both paths would otherwise share (fp8_act +
    // q8_1/d8), sized for max_batch rows. set_overlap_prefill_active: marks a
    // prefill forward enqueued concurrently with an in-flight decode; the
    // smallm/producer-xq dispatch paths decline for it (scratch+tags are decode-owned under overlap).
    bool allocate_decode_qscratch(int max_batch);
    void set_overlap_prefill_active(bool v) { overlap_prefill_active_ = v; }

    // Takes the small-M NVFP4 workspace + activation scratch for the largest
    // eligible weight from the T2 arena before the first captured decode step:
    // a capture cannot allocate, and on a GGUF source no eager forward reaches
    // the small-M block first, so without this every captured batched-decode
    // graph baked in the dequant fallback (#1897). Falls back to lazy cudaMalloc growth otherwise.
    void allocate_smallm_scratch(cudaStream_t stream);
    // The prefill sample must not land in the decode batch's parity slots
    // while both run concurrently: the engine points the prefill-side
    // samplers at a dedicated slot for the duration of an overlap prefill.
    void set_sample_slot_override(int32_t* slot) { sample_slot_override_ = slot; }
    bool has_gguf_nvfp4_overlay() const;
    bool model_has_moe() const { return has_moe_; }

    // Batched-decode residual accumulation eligibility (gemm.nvfp4_residual_beta1):
    // the o/down/GDN-out projection may run beta=1 into hidden when it will
    // take the smallm accumulate path. Shared by three call sites so the gates cannot drift apart.
    bool residual_beta1_nvfp4_ok_(TensorID id, int n, const Tensor& h) const {
        if (!dispatch_policy().gemm.nvfp4_residual_beta1 || id == kInvalidTensorID)
            return false;
        if (n <= 1 || n > 32 || h.qtype != QType::F16)
            return false;
        if (cur_spec_verify_ || overlap_prefill_active_ || lora_ != nullptr)
            return false;
        return registry_.handle(id).primary_tier == StorageTier::CUTLASS_NVFP4;
    }

    // Capacity of the [n_heads, attn_seq, attn_seq] FP16 attn-scores workspace.
    // Engine's chunked-prefill path must clamp chunk_len so n × ctx_len ≤ cap².
    // Returns 0 if the buffer wasn't allocated (VRAM-constrained / WMMA fallback).
    int attn_scores_cap() const {
        return attn_scores_buf_ ? static_cast<int>(attn_scores_.shape[1]) : 0;
    }

    // Attention shapes are uniform when per-layer shape arrays are absent or
    // carry one distinct nonzero value (GDN/Mamba2 hybrids zero non-attention
    // layers). Uniform models are servable by the O(n) FA2/FMHA family; only
    // truly heterogeneous shapes (Gemma-4 dual head_dim 256/512) need the rectangular cuBLAS path.
    bool attn_shapes_uniform() const;

    // True when FP16-QK FA2 serves ALL prefill for this model (uniform shapes,
    // no learned sinks, hd=128 always / 256 behind fa2_hd256, fa2_fp16qk !=
    // "never"). Shared by the S-matrix allocator skip and workspace_estimate() so the two cannot drift
    // (#943).
    bool fa2_serves_all_prefill() const;

    // Largest prefill chunk at `offset` the chunked-attention dispatch can
    // serve without overflowing the cuBLAS S-matrix: n*(offset+n) <= s_cap^2
    // and n <= s_cap, floored to a kv_bs multiple. O(n) FA2/FMHA-served chunks
    // need no S-matrix and return `desired` unchanged. 0 = even a kv_bs chunk
    // cannot be served; caller must reject the request.
    int max_safe_prefill_chunk(int offset, int desired, int kv_bs) const;

    // Graph-captured verify chunk (#847): the chunked continuation forward is
    // replayable only when FP16-QK FA2 serves EVERY attention layer with
    // device-read lengths: uniform hd=128, no learned sinks, no MLA, no
    // LongRoPE, fa2_fp16qk not disabled.
    bool chunk_capture_supported() const;
    // Persistent K/V scratch for the replayable chunked continuation
    // ([ctx_capacity,nkv,hd] FP16 each), replacing the per-layer
    // cudaMallocAsync whose size would bake the growing ctx_len into the
    // graph. Idempotent; false on allocation failure.
    [[nodiscard]] bool ensure_chunk_capture_scratch(int ctx_capacity);

    // Grow-once scratch for the speculative verify path (argmax partials,
    // per-vocab penalty counts), sized from init-time constants so the engine
    // pre-warms it instead of letting the first verify step allocate while serving.
    [[nodiscard]] bool ensure_verify_scratch(bool with_penalties);
    void prewarm_verify_scratch();
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
    float* h_logits_pinned() const { return h_logits_pinned_.as<float>(); }

    // Ensure pinned logits buffer is allocated for the given number of floats.
    // For single sequence: pass vocab_size. For batched logprobs: pass vocab_size * n_sequences.
    void ensure_logits_pinned(int total_floats);

    // Configure StreamingLLM smart KV cache: keeps the first n_sinks tokens
    // and the last `window` tokens, drops the rest; n_sinks=0 disables.
    // Honoured only by the FP16 GQA decode kernel; quantized variants ignore this and fall back to plain
    // sliding-window attention.
    void set_streaming_kv(int n_sinks, int window) {
        streaming_n_sinks_ = (n_sinks > 0) ? n_sinks : 0;
        streaming_window_ = (window > 0) ? window : 0;
    }
    int streaming_window() const { return streaming_window_; }

    // LoRA runtime delta (#522): activation-path low-rank deltas, no weight
    // patching, works with every quant tier. nullptr = base model. Caller
    // (Engine) must invalidate decode graphs around swaps: the captured graph
    // holds the adapter's kernel launches/pointers.
    void set_lora(const LoraAdapter* adapter);
    const LoraAdapter* lora() const { return lora_; }

    // Public view_tokens wrapper for external callers.
    Tensor view_hidden(int n_tokens) const { return view_tokens(hidden_, n_tokens); }

    // Executor reads dispatch decisions from a DispatchPolicy owned by Engine
    // (the nine former RuntimeConfig sections; RuntimeConfig::current() is
    // gone). Engine wires this via set_dispatch_policy() during init; contract
    // is "set before first access". A bare GraphExecutor in tests must wire one itself.
    void set_dispatch_policy(const DispatchPolicy& p) noexcept { dispatch_policy_ = &p; }
    // The KV cache's real block size, resolved by the engine before init().
    // Workspace sizing that converts a token count into a block count needs
    // this and not kKVBlockSize - the two differ on n_kv_heads <= 4 models.
    void set_kv_block_size(int n) noexcept { kv_block_size_ = n > 0 ? n : kKVBlockSize; }

    // Activation calibration ([calibration] enabled): collects per-input-
    // channel activation magnitudes off gemm_via_handle_ for an offline
    // quantizer. Engine turns this on before the first forward and turns CUDA
    // graphs off with it, since the collector allocates lazily (a capture forbids that).
    void enable_calibration() {
        if (!calib_)
            calib_ = std::make_unique<ActivationCalibrator>(vram_alloc_);
    }
    const ActivationCalibrator* calibration() const { return calib_.get(); }

    const DispatchPolicy& dispatch_policy() const noexcept {
        // CRITICAL: set_dispatch_policy() must be called before any forward.
        // Hard-failing here would crash unit tests; cold default is acceptable.
        static const DispatchPolicy kDefault;
        return dispatch_policy_ ? *dispatch_policy_ : kDefault;
    }

private:
    // The init-time weight-quantization pipeline (23 pre_dequant_*/
    // nvfp4_decode_* methods + the build-only StoragePlan) was extracted to
    // QuantPipeline (exec/quant_pipeline.h); GraphExecutor owns one and
    // delegates pre_dequant_weights() to it.

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
    // hidden_[0..n_rows) in batches of max_logit_tokens_, calling
    // consume(logits_view, row0, csz) per batch (softcap already applied).
    // Used by perplexity_nll_partial and greedy_argmax_all. allow_cutlass
    // routes NVFP4 LM heads through the CUTLASS NVFP4-activation GEMM (true
    // only for the perplexity harness); spec-decode verify passes false so
    // batch=1 output stays bit-identical to the FP16-activation GEMV.
    void for_each_lm_head_batch_(int n_rows, cudaStream_t stream, bool allow_cutlass,
                                 const std::function<void(const Tensor&, int, int)>& consume);

    // Spec-decode verify argmax partials (lazy, grows only; freed in
    // free_buffers).
    void* verify_argmax_scratch_ = nullptr;
    size_t verify_argmax_scratch_sz_ = 0;
    // Spec-decode verify penalty counts: [vocab] int32 occurrence counts of
    // the shared history (lazy; freed in free_buffers).
    int32_t* verify_pen_counts_ = nullptr;
    int verify_pen_counts_cap_ = 0;

    // Activation calibration collector — null unless [calibration] enabled.
    std::unique_ptr<ActivationCalibrator> calib_;

    // LoRA (issue #522)
    const LoraAdapter* lora_ = nullptr;
    void* lora_scratch_ = nullptr;  // fp32[max_rank] + fp16[max_tokens*max_rank]
    size_t lora_scratch_sz_ = 0;
    void lora_delta_(const LoraWeights& w, const void* x, void* y, int n, cudaStream_t stream);
    int max_logit_tokens_ = 0;     // max tokens needing LM head projection (= max(max_batch_size, 8))
    int kv_block_size_ = kKVBlockSize;  // real KV block size (set_kv_block_size)
    int cur_n_tokens_ = 0;         // set by forward_logits for use by run_ffn
    int cur_layer_ = -1;           // set by forward_logits; keys calibration entries
    int cur_decode_step_ = 0;      // set by forward_logits for debug dump tagging
    bool cur_force_fp16_ = false;  // set by forward_logits, bypasses FP8 GEMM paths
    bool cur_spec_verify_ = false; // set by forward_logits: spec-verify chunk (#998)
    // Set by forward_logits: the forward's rows are generated tokens (batched
    // decode or a spec-verify chunk), not prompt rows. Gates the GGUF NVFP4
    // decode overlay for 2..32-row GEMMs (#1897): M alone cannot tell a decode
    // step from a short prefill.
    bool cur_decode_rows_ = false;
    bool cur_per_row_lm_ = false;  // set by forward_logits, per-row Q8_1 LM head

    // Programmatic Dependent Launch: when true, custom kernels have the PDL
    // attribute set so the GPU can overlap tail of one kernel with head of next.
    bool use_pdl_ = false;

    // Owns the workspace buffers + sizes + the decode/prefill swap state; the
    // moved methods write the activation/phase tensors below through pointers
    // set in ws_.init() (see exec/workspace.h).
    Workspace ws_;

    // Persistent activation tensors (views into the persistent workspace)
    Tensor hidden_;    // [max_tokens, d_model] FP16
    Tensor residual_;  // [max_tokens, d_model] FP16
    Tensor norm_out_;  // [max_tokens, d_model] FP16
    Tensor logits_;    // [max_logit_tokens, vocab_size]

    // FP32 residual accumulator for post-norm architectures (Gemma-3):
    // prevents FP16 overflow in the residual stream over many layers. FP32 is
    // the "true" hidden state; FP16 hidden_ is only RMSNorm input (scale-
    // invariant, so clamping is safe). nullptr for pre-norm models.
    void* fp32_accum_buf_ = nullptr;
    Tensor fp32_hidden_;  // [max_tokens, d_model] FP32 — true hidden state

    // Shared/persistent workspace buffers + per-phase sizes live in ws_. The
    // phase tensors below are views carved by GraphExecutor's
    // configure_*_workspace methods (slicing ws_.shared()); the hot path reads them as members.

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

    // Persistent K/V gather scratch for the EAGER chunked path. Spec-verify
    // re-enters that path per layer per verify step; per-call
    // cudaMallocAsync/FreeAsync cost ~140 alloc pairs per verify on hybrids
    // (#847). Grow-only, reused across layers (stream-ordered use).
    half* chunk_eager_k_ = nullptr;
    half* chunk_eager_v_ = nullptr;
    size_t chunk_eager_bytes_ = 0;

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

    // Chunk-parallel GDN prefill scan workspace (gdn.chunkpar_scan): five
    // per-(chunk,head) strip arrays + FP32 inter-strip state. Engine lifetime,
    // allocated when the model has GDN layers; nullptr degrades to the fused scan.
    void* gdn_chunkpar_ws_ = nullptr;
    size_t gdn_chunkpar_ws_bytes_ = 0;

    // GDN alpha/beta narrow GEMM workspace (gdn.alpha_beta_smallm): split-K
    // partials + per-tile tickets, ~100 KiB, zeroed once at allocation.
    // nullptr keeps the two cuBLAS calls.
    void* gdn_ab_ws_ = nullptr;
    size_t gdn_ab_ws_bytes_ = 0;
    bool gdn_ab_narrow_logged_ = false;
    bool gdn_m1_input_logged_ = false;
    bool gdn_m1_out_logged_ = false;
    bool attn_m1_qkv_logged_ = false;

    // --- Separately allocated buffers (not part of unified workspace) ---

    // LRU cache for host-resident expert weights on GPU.
    // Keeps recently-used experts in VRAM to avoid repeated H2D copies.
    ExpertLRUCache expert_cache_;

    // Pre-allocated dequant scratch for the gemm_nvfp4 fallback (M>1 only).
    // True when nvfp4_dequant_ws_buf_ came from the engine-persistent arena
    // (freed wholesale) rather than from VRAMAllocator.
    bool nvfp4_dequant_ws_from_arena_ = false;
    // Set up by allocate_nvfp4_dequant_workspace() and registered with the
    // free function via set_nvfp4_dequant_workspace(). Allows the fallback
    // path to run inside CUDA stream capture without crashing on cudaMalloc.
    void* nvfp4_dequant_ws_buf_ = nullptr;
    size_t nvfp4_dequant_ws_size_ = 0;

    // MLA persistent QKV scratch, pre-allocated once (sized for max_tokens) so
    // the materialized two-step KV projection never calls cudaMallocAsync
    // inside the CUDA-graph-captured decode region (capture rejects
    // stream-ordered alloc/free -> silent eager fallback + degeneration).
    void* mla_kv_a_buf_ = nullptr;    // [max_tokens, kv_lora_rank + qk_rope_head_dim]
    void* mla_latent_buf_ = nullptr;  // [max_tokens, kv_lora_rank]
    void* mla_k_rope_buf_ = nullptr;  // [max_tokens, qk_rope_head_dim]
    void* mla_kv_b_buf_ = nullptr;    // [max_tokens, n_heads*(qk_nope_head_dim+v_head_dim)]

    // MLA absorbed-decode latent KV cache (opt-in attention.mla_absorb).
    // Per-layer slice [max_seq, kv_lora_rank+qk_rope_head_dim] FP16: cols
    // [0:kv_lora_rank] = RMSNorm'd latent, rest = post-RoPE decoupled key.
    // Allocated only when mla_absorb is set, model is_mla(), every layer's kv_b_proj is FP16. Single-sequence
    // only.
    void* mla_absorb_cache_ = nullptr;        // [n_layers, max_seq, kv_lora+rope]
    float* mla_absorb_scores_ = nullptr;      // [n_heads, max_seq] decode scratch
    size_t mla_absorb_layer_stride_ = 0;      // halfs per layer = max_seq*(kv_lora+rope)
    int mla_absorb_max_seq_ = 0;
    // The MLA QKV quartet is the one T2 tenant with no null-tolerant consumer, so
    // an arena that cannot serve it fails the load instead of handing out a
    // pointer the KV projection will dereference (A7 step 4b.2, I6).
    bool mla_scratch_unservable_ = false;

    // Set when allocate_nvfp4_dequant_workspace() could NOT pre-allocate the
    // M>1 dequant scratch (weight exceeds the cap, or alloc failed). The
    // fallback then lazy-cudaMallocs, illegal inside CUDA graph capture; the
    // scheduler reads this to skip prefill-graph capture.
    bool nvfp4_dequant_uncapturable_ = false;

public:
    bool nvfp4_dequant_uncapturable() const { return nvfp4_dequant_uncapturable_; }

    // True when MoE prefill will run the legacy host-args fallback (no CUTLASS
    // NVFP4 grouped workspace, e.g. GGUF Q*_K MoE). That path reads routing on
    // the host (D2H+sync) and throws under active capture, so the scheduler
    // must not attempt prefill-graph capture at all (#874).
    bool moe_prefill_uncapturable() const;

private:
    // Init-time weight-quantization pipeline (D2 extraction). Owns the build-only
    // StoragePlan; fills the long-lived caches below by reference in build().
    QuantPipeline quant_pipeline_;

    // Weight caches (FP16, FP8, NVFP4, CUTLASS NVFP4/MXFP4, fused KV/gate+up)
    WeightCaches wcache_;

    // Mode flags mirrored from wcache_ for PlanHints (hints_ is the Phase 5 source of truth).
    PlanHints hints_;

    // The build-only StoragePlan (storage_plan_) and apply_arch_rules_()
    // moved into QuantPipeline (exec/quant_pipeline.h) — no hot-path reader.

    // WeightRegistry: parallel handle store (Phase 2+ shim, populated alongside wcache_)
    WeightRegistry registry_;

    // Quantization scratch buffers (FP8 act, CUTLASS act, dp4a, dequant, split-K)
    QuantScratch qscratch_;
    // Per-slot decode copies of the shared quant scratches (overlap only;
    // null when allocate_decode_qscratch never ran). use_workspace swaps the
    // seven fields between qscratch_ and the saved prefill values.
    QuantScratch qscratch_decode_;
    bool qscratch_decode_ready_ = false;
    struct {
        void* fp8_act; size_t fp8_act_size; float* d_act_scale; float* d_fp8_block_maxes;
        float* d_fp8_absmax; int fp8_max_grid; void* q8_1_buf; float* d8_buf; int q8_1_rows;
    } qscratch_prefill_save_{};
    bool overlap_prefill_active_ = false;
    int32_t* sample_slot_override_ = nullptr;

    // CUTLASS NVFP4 LM head for batched-decode (n>1) tensor-core GEMM. Borrows
    // the FP4 data from the NVFP4 decode cache; owns only the SfAtom scales.
    // Built by build_lm_head_cutlass_() when serving; freed in the destructor.
    CutlassNvFP4Weight lm_head_cutlass_{};
    bool lm_head_cutlass_ready_ = false;

    // Pre-allocated sampling result buffers (avoids cudaMalloc/cudaFree per token).
    void apply_row_filters_(float* lp, int vocab, const InferenceState& state, cudaStream_t stream);
    const int32_t* banned_cache_(const InferenceState& state, cudaStream_t stream);
    void flush_pending_topk_rows_(cudaStream_t stream);
    // Row-batched top-k/top-p staging: sample_single_from_logits_async stashes
    // eligible rows here instead of launching; collect_sampled_tokens uploads
    // args (one pinned H2D) and fires ONE partial + ONE finalize launch for
    // the whole batch. Greedy rows launch immediately.
    PinnedBuffer h_row_args_;            // pinned, 2 x sample_slots_ entries (parity halves, T5b)
    TopkRowArgs* d_row_args_ = nullptr;  // device mirror (same layout)
    int n_pending_topk_rows_ = 0;
    int pending_topk_max_k_ = 0;
    int pending_topk_vocab_ = 0;
    // Greedy + penalty row staging, same stash/flush contract as top-k above.
    // Flush order in the collectors is penalties -> greedy -> top-k, preserving
    // each row's own penalties-before-sampler order on the single stream.
    void flush_pending_greedy_rows_(cudaStream_t stream);
    void flush_pending_penalty_rows_(cudaStream_t stream);
    PinnedBuffer h_greedy_args_;              // pinned, 2 x sample_slots_ (parity halves)
    GreedyRowArgs* d_greedy_args_ = nullptr;  // device mirror
    int n_pending_greedy_rows_ = 0;
    PinnedBuffer h_pen_args_;                 // pinned, 2 x sample_slots_ (parity halves)
    PenaltyRowArgs* d_pen_args_ = nullptr;    // device mirror
    int n_pending_pen_rows_ = 0;
    int pending_sample_vocab_ = 0;
    // Static banned-token list cache (see apply_row_filters_): keyed on the
    // host pointer + count, freed in free_buffers.
    int32_t* d_banned_cache_ = nullptr;
    const int32_t* banned_cache_src_ = nullptr;
    int banned_cache_n_ = 0;
    size_t banned_cache_capacity_ = 0;
    int32_t* d_sample_result_ = nullptr;  // device buffer for argmax/sample kernel output
                                          // (2 x sample_slots_ x SAMPLE_SCRATCH_BYTES parity
                                          // halves; slot 0 of parity 0 = single-seq)
    PinnedBuffer h_sample_pinned_;        // async D2H sample result staging (T5b)
                                          // (2 x sample_slots_ int32s, parity halves)
    int sample_slots_ = 0;                // batched sampling slots per parity set (= max_batch_size)
    int sample_parity_ = 0;               // active parity half for async enqueue/flush/gather
    cudaEvent_t sample_gather_evt_[2] = {nullptr, nullptr};  // per-parity gather-done events

    // Pinned host buffer for logprobs extraction (D2H copy of logits)
    PinnedBuffer h_logits_pinned_;      // [vocab_size] pinned host memory (T5b)
    int h_logits_pinned_size_ = 0;      // vocab_size used for allocation

    // --- Layer index mappings ---

    // Mapping from global layer index to SSM layer index (for SSMState access)
    std::vector<int> ssm_layer_map_;  // ssm_layer_map_[global_idx] = ssm_idx, or -1

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

    // Columns of the SSM z buffer, which the attention output gate borrows
    // (exec_ssm_z_cols). Cached at init: configure_ssm_workspace() runs on every
    // recurrent layer of every forward, and computing it scans all layers.
    int ssm_z_cols_ = 0;

    // The dual prefill/decode workspace swap state (decode_workspace_,
    // decode_shared_workspace_, the sizes, decode_max_batch_, active_workspace_,
    // SavedWorkspace saved_prefill_ws_) moved into ws_ (exec/workspace.h).

    // --- Layer offload manager (non-owning, set by engine) ---
    LayerOffloadManager* offload_mgr_ = nullptr;

    // --- Per-Engine RuntimeConfig (Phase 5 Track D, non-owning) ---
    // Engine wires this via set_dispatch_policy() during Engine::init.
    // Replaces RuntimeConfig::current() inside GraphExecutor::* methods.
    const DispatchPolicy* dispatch_policy_ = nullptr;

    // The shared/persistent/decode scratch arena (allocate_*_workspace,
    // compute_shared_sizes, workspace_estimate, resize_workspace,
    // allocate_decode_workspace, use_workspace) moved into Workspace
    // (exec/workspace.h); GraphExecutor owns ws_ and delegates. The four
    // per-phase carvers stay here: they write GraphExecutor's activation-tensor
    // members by slicing the shared buffer, so they belong on GraphExecutor.
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
    // Sibling-pair small-M dispatch: two weights consuming the SAME input,
    // both routing to the smallm v2 kernel, run as one launch
    // (gemm_nvfp4_smallm_v2_pair_a4). False (and no-op) when either weight
    // would not take that route; caller then issues the two separate calls it would have anyway.
    bool try_smallm_pair_dispatch_(TensorID id_a, TensorID id_b, const Tensor& input,
                                   Tensor& out_a, Tensor& out_b, const GemmContext& ctx);
    // Batched-decode GDN alpha+beta projections in one narrow FP16 launch
    // (gdn.alpha_beta_smallm). False and no-op when a weight is not
    // FP16-resident or a shape doesn't fit; caller then issues its two calls.
    bool try_gdn_alpha_beta_narrow_(TensorID alpha_id, TensorID beta_id, const Tensor& input, Tensor& alpha_out,
                                    Tensor& beta_out, cudaStream_t stream);
    // The NVFP4 view gemm_via_handle_ would hand the M=1 decode GEMV for this
    // weight (decode tier NVFP4 in wcache_.nvfp4, or native CUTLASS_NVFP4 via
    // its source bytes). false = that dispatch would not take the GEMV.
    bool nvfp4_decode_weight_(TensorID id, NvFP4QuantResult& out) const;
    // M=1 GDN launch fusion (gdn.m1_fused): in_proj+gate+alpha+beta in one
    // GEMV; out-proj adds the residual in its epilogue. False and no-op when a
    // weight/shape doesn't fit. alpha/beta land in ssm_dt_buf_ (alpha at 0,
    // beta at the 256-byte-aligned offset) for the scan to read.
    bool try_gdn_input_fused_m1_(const TransformerLayer& ly, const Tensor& input, Tensor& proj, Tensor& gate_out,
                                 int n_heads, cudaStream_t stream);
    bool gdn_out_residual_m1_ok_(const TransformerLayer& ly, int n, const Tensor& h) const;
    void gdn_out_residual_m1_(const TransformerLayer& ly, const Tensor& y, Tensor& h, cudaStream_t stream);
    // M=1 attention q|k|v in one NVFP4 GEMV launch on gated-attention models
    // (the ungated nvfp4_qkv path already does this). Same decline contract.
    bool try_attn_qkv_fused_m1_(const TransformerLayer& ly, const Tensor& input, Tensor& q_out, Tensor& k_out,
                                Tensor& v_out, cudaStream_t stream);
    // General form: 2..3 weights on one input (attention q|k|v adds the
    // striped k/v shapes to q's single-stripe wave). Outputs must have row
    // stride N.
    bool try_smallm_multi_dispatch_(const TensorID* ids, Tensor* const* outs, int count, const Tensor& input,
                                    const GemmContext& ctx);
    // True when an M>1 dispatch of `id` is guaranteed to take the CUTLASS
    // NVFP4 prefill block (which quantizes input into shared activation
    // scratch). Gates the act-quant-hint dedupe at QKV/gate-up call sites;
    // must be CONSERVATIVE: false negative just re-quantizes, false positive skips with stale scratch.
    bool prefill_routes_cutlass_nvfp4_(TensorID id, int M) const;
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
    // Cheap precondition mirror for the device-args fast path, read upstream
    // of moe_gather to skip the gather when the path is guaranteed to fire (it
    // reads ctx.no via sorted_token_ids and doesn't need the gathered
    // intermediate). A mismatch costs at most one wasted gather, never wrong output.
    bool moe_cutlass3x_will_use_device_args_(int layer, const MoeFfnContext& ctx) const;
    // Optional shared expert (parallel dense FFN), called from run_moe_ffn
    // after routed experts wrote into h. Reads `no` (post-norm), adds its
    // result into `h` via elementwise_add. No-op when ly.w_up_shared is null
    // or the runtime opt-out is set.
    void run_shared_expert_ffn(int layer, cudaStream_t stream, int n, int d,
                               float eps, const Tensor& no, Tensor& h);
    // MoE decode fast-path (n=1, device-resident packed experts): dispatches
    // all top_k experts in a single kernel per projection (NVFP4 and dp4a/FP16
    // sub-paths handled internally). Sets residual_fused=true when the
    // weighted sum fused the residual add (no shared expert active).
    void run_moe_decode_fast(int layer, cudaStream_t stream, int n, int d, int eff,
                             int top_k, const MoeRoutingResult& routing,
                             const Tensor& no, Tensor& h, const Tensor& r,
                             bool moe_use_fp32_residual, bool moe_fused_norm_q8,
                             bool will_skip_residual_copy, bool& residual_fused);
    // Decode step for a layer whose NVFP4 experts live on host: stages routed
    // experts into the LRU cache's slot pool, runs ordinary fused NVFP4 GEMVs
    // against it. Called only when run_moe_decode_fast's dispatch predicate
    // (nvfp4_expert_offload.h) confirms every precondition holds.
    void run_moe_decode_nvfp4_host(int layer, cudaStream_t stream, int d, int eff, int top_k,
                                   const MoeRoutingResult& routing, const Tensor& no, Tensor& h,
                                   const Tensor& r, bool moe_use_fp32_residual,
                                   bool will_skip_residual_copy, bool& residual_fused,
                                   bool non_gated_experts);

    // Stages every expert of one host-resident NVFP4 layer into
    // moe_.layer_stage_buf, so prefill reads them from device memory instead
    // of two H2D per expert. Fills `out` with one view per projection; true when at least one was staged.
    bool stage_nvfp4_layer_(int layer, cudaStream_t stream, StagedProj out[kExpertProjCount]);

    // Builds a CUTLASS device-args view over a layer staged by
    // stage_nvfp4_layer_, so a host-resident layer dispatches like a
    // device-resident one. False when the layer was not staged, a projection
    // could not build its SfAtom view, or the opt-in is off.
    bool build_staged_device_args_(const MoeFfnContext& ctx, bool non_gated,
                                   MoEWorkspace::PerLayerNvfp4DeviceArgsCache& out) const;

    // Stage a host-resident layer (once per MoE call, carried on ctx) and say
    // whether the staged copy can carry the CUTLASS prefill.
    bool stage_layer_for_prefill_(int layer, cudaStream_t stream, MoeFfnContext& ctx);

public:
    // Throws when a MoE layer's NVFP4 experts are host-resident and nothing
    // can serve them from there. Call after pre_dequant_weights(): needs
    // Phase 0's promotion and the initialised expert cache.
    void verify_host_expert_placement() const;

private:
    // Computes MoE routing: gate logits (FP32 router fast-path for Gemma-4
    // already done by caller, signaled via fp32_gate_logits_ready) + topk
    // gating + per-expert weight scaling (Nemotron, Gemma-4). Caller passes
    // pre-normalized router_in if !fp32_gate_logits_ready.
    void compute_moe_routing(int layer, cudaStream_t stream, int n, int d, int ne,
                             int top_k, const Tensor& router_in,
                             bool fp32_gate_logits_ready, bool will_decode_fast,
                             const void* router_bias_ptr, bool use_sigmoid,
                             bool norm_weights, MoeRoutingResult& routing);
public:
    // Per-layer expert imbalance, readable while serving (#1548). peak = worst
    // max(M_e) on that layer; mean_max = average per-launch maxima; mean_rows
    // = average rows per expert per launch. mean_max/mean_rows says whether a
    // layer is padding-bound (grouped GEMM pads every expert to one M tile).
    struct MoeImbalance {
        uint32_t peak_max = 0;
        double mean_max = 0.0;
        double mean_rows = 0.0;
        uint32_t launches = 0;
    };
    std::vector<MoeImbalance> moe_imbalance() const;

private:
    // Write + release the expert-activation histogram (diagnostics.moe_expert_hist).
    // Destructor-time, so it covers the whole process rather than one request.
    void dump_moe_expert_hist_();
    void dump_moe_expert_trace_();
    // Fused Q6_K prefill MoE path: reads Q6_K weights directly, no FP16
    // dequant scratch. TC variant uses gather-free sorted_token_ids
    // indirection; scalar variant materializes the gathered buffer. Fills
    // moe_.expert_{gate,up,swiglu,down}; true when the path was taken.
    bool try_run_moe_q6k_prefill(int layer, cudaStream_t stream, int n, int d, int eff,
                                 int ne, int expanded, bool non_gated_experts, QType up_qtype,
                                 const MoeRoutingResult& routing, const Tensor& no);
    // Fused Q4_K prefill: reads Q4_K weights directly, FP16 activations from
    // L1/L2 cache. Same interface as Q6_K but with Q4_K dequant logic.
    bool try_run_moe_q4k_prefill(int layer, cudaStream_t stream, int n, int d, int eff,
                                 int ne, int expanded, bool non_gated_experts, QType up_qtype,
                                 const MoeRoutingResult& routing, const Tensor& no);
    // Gemma-4 ggml MMVQ per-token prefill: FP16 batch dequant + a single
    // cublasGemmGroupedBatchedEx per projection (dequants all experts in one
    // shot). One D2H sync per layer for offsets (unavoidable for the grouped
    // GEMM API). Falls through to scatter; true when the path was taken.
    bool try_run_moe_fp16_batch_prefill(int layer, cudaStream_t stream, int n, int d, int eff, int ne,
                                        int expanded, bool non_gated_experts, QType up_qtype,
                                        const MoeRoutingResult& routing);
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

    // Writes computed K/V into KV cache blocks. Default call (row_begin=0,
    // n_rows=-1, null overrides) is the historical single-sequence path. The
    // ragged-prefill per-seq loop passes a row range plus FLAT per-seq block
    // tables and a positions pointer, forcing single-sequence kernel indexing.
    void write_kv_cache(int layer, const InferenceState& state, cudaStream_t stream, int row_begin = 0,
                        int n_rows = -1, const int* bt_flat = nullptr, const int* bt_swa_flat = nullptr,
                        const int* positions_override = nullptr);

    // Create a Tensor view of the first n_tokens rows of a max_tokens buffer.
    Tensor view_tokens(const Tensor& buf, int n_tokens) const;
};

}  // namespace imp
