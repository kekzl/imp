#pragma once

#include "model/model.h"
#include "model/chat_template.h"
#include "runtime/scheduler.h"
#include "runtime/request.h"
#include "runtime/batch.h"
#include "runtime/green_ctx.h"
#include "runtime/cuda_graph.h"
#include "runtime/vision_pipeline.h"
#include "runtime/constraint_manager.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "memory/ssm_state.h"
#include "memory/gdn_state.h"
#include "memory/layer_offload.h"
#include "memory/vram_allocator.h"
#include "graph/executor.h"
#include "core/cuda_raii.h"
#include <memory>
#include <string>
#include <cuda_runtime.h>

namespace imp {

struct EngineConfig {
    int max_batch_size = 0;       // 0 = auto (engine detects from model size vs VRAM)
    int max_seq_len = 0;          // 0 = auto (engine detects from model metadata + VRAM)
    int kv_cache_max_blocks = 0;  // 0 = auto
    bool use_green_contexts = false;
    float green_ctx_prefill_ratio = 0.8f;
    bool use_cuda_graphs = true;
    bool use_pdl = true;
    QType compute_dtype = QType::F16;

    // Default sampling parameters
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = -1;

    // KV cache dtype: FP16 (default) or FP8_E4M3 for ~50% KV VRAM savings
    QType kv_cache_dtype = QType::F16;

    // TurboQuant Lite: sketch_dim = turboquant_sketch_multiplier * head_dim
    int turboquant_sketch_multiplier = 2;

    // SSM state dtype: FP32 (default) or FP16 for ~50% VRAM savings on h_state
    QType ssm_state_dtype = QType::F32;

    // VRAM budget: max GPU memory to use (MiB), 0 = use all available
    size_t vram_budget_mb = 0;

    // Layer offloading: number of layers to keep on GPU (-1 = all on GPU, 0 = all offloaded)
    int gpu_layers = -1;

    // KV cache block size (tokens per block). 0 = auto-select based on model.
    // Larger blocks (32, 64) improve coalescing for GQA models with few KV heads.
    int kv_block_size = 0;

    // Chunked prefill
    int prefill_chunk_size = -1;  // -1 = per-arch default (512 if supported, 0 otherwise)

    // FP8 prefill weight cache: uses FP8 E4M3 instead of FP16 for ~2x prefill throughput
    bool use_fp8_prefill = false;

    // NVFP4 decode weight cache: -1=auto, 0=off, 1=additive (FP16+NVFP4), 2=NVFP4 only
    int use_nvfp4_decode = -1;

    // Minimum KV cache tokens. Budget planner guarantees at least this many
    // tokens of KV capacity before allocating weight caches. 0 = auto.
    int min_kv_tokens = 0;

    // MXFP4 prefill: CUTLASS MXFP4 GEMM for prefill (converts NVFP4 → MXFP4 format, sm_120)
    bool use_mxfp4_prefill = false;

    // Dual-path quantization: attention weights (WQ/WK/WV/WO) stay at FP8 for quality,
    // FFN weights (gate/up/down) use NVFP4 for 2x bandwidth reduction during decode.
    bool dual_path_quant = false;

    // Prefix caching: reuse KV cache blocks for shared token prefixes
    bool use_prefix_caching = false;
    std::string prefix_cache_path;  // path to persist prefix cache (empty = disabled)

    // Vision (multimodal)
    std::string mmproj_path;  // path to mmproj GGUF, empty = text-only

    // StreamingLLM smart KV cache (Xiao et al., 2023): keep first N "sink"
    // tokens + last W tokens, drop everything in between. Reduces decode KV
    // bandwidth and enables long-running generations without VRAM blowup.
    // Currently active only on the FP16 GQA decode path; quantized variants
    // ignore these settings.
    bool streaming_kv_enabled = false;
    int streaming_kv_n_sinks = 4;    // # initial tokens to always attend
    int streaming_kv_window = 0;     // 0 = derive from ModelConfig::sliding_window
    int streaming_kv_threshold = 0;  // 0 = auto: n_sinks + window + 2*kKVBlockSize
};

class Engine {
public:
    Engine() = default;
    ~Engine();

    [[nodiscard]] bool init(std::shared_ptr<Model> model, const EngineConfig& config);

    // Run one step of inference (prefill or decode depending on scheduler)
    [[nodiscard]] bool step();

    // High-level generate with sampling parameters
    std::string generate(const std::string& prompt, int max_tokens, float temperature = 1.0f,
                         float top_p = 1.0f, int top_k = 0, int seed = -1, bool apply_chat_template = true,
                         float min_p = 0.0f, float repetition_penalty = 1.0f, float frequency_penalty = 0.0f,
                         float presence_penalty = 0.0f);

    void add_request(std::shared_ptr<Request> req);

    // Reset SSM state for a sequence (call on context_reset for hybrid models)
    void reset_ssm_state(int seq_id);

    // Invalidate all cached CUDA graphs (call on context_reset to ensure
    // deterministic output — stale graph captures can produce different results)
    void invalidate_graphs();

    // Reset batch pool upload cache (call on context_reset to prevent
    // stale block table pointers when KV blocks are reused)
    void reset_batch_pool_cache();

    // Vision: set image for next generation. Returns false if no mmproj loaded.
    [[nodiscard]] bool set_image(const std::string& path);
    [[nodiscard]] bool set_image_from_memory(const uint8_t* data, size_t len);
    void clear_image();
    bool has_vision() const noexcept { return vision_.is_available(); }
    bool has_vision_input() const noexcept { return vision_.has_input(); }

    // Enable MTP-based speculative decoding. K = draft length (1-4 typical).
    // Requires model->mtp_->loaded. Allocates the MTP workspace via the VRAM
    // allocator. Returns false if MTP head not present or workspace alloc fails.
    // Phase 3 scaffolding: API in place; actual draft-verify loop integration
    // is Phase 4 work (auto-invoke from decode path).
    bool enable_mtp_spec_decode(int k);
    bool mtp_spec_decode_enabled() const noexcept { return mtp_spec_k_ > 0; }
    int  mtp_spec_decode_k() const noexcept { return mtp_spec_k_; }

    // One draft step. Public for Phase 5 smoke testing; production callers
    // should not invoke directly until Phase 4 wires this into the decode loop.
    bool mtp_draft_one(int prev_token_id, const void* d_h_prev,
                       int hidden_dim, int vocab_size, int* out_token_id);

    // Phase 3.5 telemetry: tracks "what fraction of decode-step next-tokens
    // would the MTP head have correctly predicted from the previous step?"
    // Populated automatically by step_decode when mtp_spec_decode_enabled()
    // && single-sequence batches. Does NOT change generation — the actual
    // next_token still comes from the main forward+sample. Provides the
    // measurement Phase 5.5 needs to decide whether a batched-verify
    // implementation of Phase 3.5 is ROI-worthy.
    struct MtpAccuracy {
        int matches = 0;
        int total   = 0;
        float rate() const { return total > 0 ? static_cast<float>(matches) / total : 0.0f; }
    };
    MtpAccuracy mtp_accuracy() const noexcept { return mtp_accuracy_; }
    void mtp_accuracy_reset() noexcept {
        mtp_accuracy_ = {};
        mtp_pending_prediction_ = -1;
    }

    // Accessors for C API
    Scheduler* scheduler() const noexcept { return scheduler_.get(); }
    KVCacheManager* kv_manager() const noexcept { return kv_manager_.get(); }
    KVCache* kv_cache() const noexcept { return kv_cache_raw_; }
    Model* model() const noexcept { return model_.get(); }
    const ChatTemplate& chat_template() const noexcept { return chat_template_; }
    const std::vector<int32_t>& banned_token_ids() const { return banned_token_ids_; }
    GraphExecutor* executor() const noexcept { return executor_.get(); }
    VRAMAllocator& vram_allocator() noexcept { return vram_alloc_; }

private:
    // ── Core components ──────────────────────────────────────────────
    VRAMAllocator vram_alloc_;
    std::shared_ptr<Model> model_;
    EngineConfig config_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<KVCacheManager> kv_manager_;
    KVCache* kv_cache_raw_ = nullptr;  // Non-owning pointer (owned by kv_manager_)
    std::unique_ptr<GraphExecutor> executor_;
    GreenContextManager green_ctx_;
    CudaStream stream_;
    CudaEvent prefill_done_;
    CudaEvent decode_done_;
    int next_request_id_ = 0;

    // ── Decode batching ──────────────────────────────────────────────
    GPUBatchPool decode_batch_pool_;
    BatchBuilder decode_builder_;
    std::vector<int> padded_block_table_;
    std::vector<std::shared_ptr<Request>> sched_prefill_batch_;
    std::vector<std::shared_ptr<Request>> sched_decode_batch_;
    std::vector<std::shared_ptr<Request>> valid_decode_;

    // ── CUDA Graphs ──────────────────────────────────────────────────
    // Per-batch-size graph pool: avoids re-capture when batch size changes
    // during continuous batching (key = n_sequences).
    static constexpr int kMaxGraphPoolSize = 32;
    CudaGraphRunner decode_graph_pool_[kMaxGraphPoolSize];  // index = n_sequences - 1
    int last_decode_max_blocks_per_graph_[kMaxGraphPoolSize] = {};
    int32_t* h_sample_pinned_ = nullptr;
    // Async conditional graph loop
    CudaGraphConditionalRunner async_graph_runner_;
    std::shared_ptr<Request> async_graph_req_;
    int* async_d_block_tables_ = nullptr;
    int32_t* async_d_banned_tokens_ = nullptr;
    std::vector<int32_t> async_pending_tokens_;
    int async_pending_cursor_ = 0;

    // ── Model-specific state ─────────────────────────────────────────
    std::unique_ptr<SSMState> ssm_state_;
    std::unique_ptr<GDNState> gdn_state_;
    bool has_pure_ssm_layers_ = false;  // true if model has Mamba2 SSM layers (not GDN)
    std::unique_ptr<LayerOffloadManager> offload_mgr_;
    bool experts_on_host_ = false;
    bool dequant_done_ = false;
    ChatTemplate chat_template_;

    // ── Extracted subsystems ─────────────────────────────────────────
    VisionPipeline vision_;
    ConstraintManager constraints_;

    // ── Pre-allocated prefill metadata (eliminates per-request cudaMalloc) ──
    void* prefill_pool_ = nullptr;
    size_t prefill_pool_size_ = 0;
    int32_t* d_pf_token_ids_ = nullptr;
    int* d_pf_positions_ = nullptr;
    int* d_pf_block_tables_ = nullptr;
    int* d_pf_context_lens_ = nullptr;
    int* h_pf_positions_ = nullptr;      // pinned host staging
    int32_t* h_pf_token_ids_ = nullptr;  // pinned host staging

    // ── Penalty token buffer ─────────────────────────────────────────
    int32_t* d_penalty_tokens_ = nullptr;
    size_t d_penalty_tokens_capacity_ = 0;

    // ── BitDecoding Phase 3 residual metadata (per-step) ─────────────
    // residual_meta_d_buf_ is the legacy multi-seq metadata buffer
    // (cudaMallocAsync per step in step_decode_continuous). d_kv_slot_buf_ is
    // a persistent [max_batch_size] device array of slot indices indexed by
    // batch position, updated lazily via cudaMemcpyAsync when the batch
    // composition changes — graph-capture-safe (kernels read from a stable
    // device pointer; host updates between graph replays).
    int* residual_meta_d_buf_ = nullptr;
    int* d_kv_slot_buf_ = nullptr;
    std::vector<int> residual_meta_h_seq_ids_;
    std::vector<int> residual_meta_h_slots_;
    std::vector<int> residual_meta_h_counts_;
    std::vector<int> residual_meta_h_widxes_;
    // Last-uploaded slot per batch position; only re-upload when changed.
    std::vector<int> d_kv_slot_last_uploaded_;

    // ── MTP spec-decode (Phase 3 scaffolding) ──────────────────────
    // Workspace + draft length for MTP-driven speculative decoding. Active
    // when mtp_spec_k_ > 0 AND the loaded model has model->mtp_->loaded.
    // Phase 3: API in place (enable_mtp_spec_decode + mtp_draft_step), NOT
    // yet auto-invoked by the decode loop. Phase 4 wires CLI flag, Phase 5
    // measures acceptance.
    // Defined in <runtime/mtp_forward.h>; forward-declared to avoid include.
    int mtp_spec_k_ = 0;
    void* mtp_ws_storage_ = nullptr;  // type-erased MtpDraftWorkspace*

    // Phase 3.5 telemetry: rolling MTP-draft-accuracy across the active session.
    // mtp_pending_prediction_ is the prediction made at the end of the
    // PREVIOUS decode step; it gets compared to the actual next_token at the
    // start of the CURRENT step. -1 = no pending prediction (start of session,
    // batch>1, prediction call failed, etc).
    int mtp_pending_prediction_ = -1;
    MtpAccuracy mtp_accuracy_{};

    // ── Banned tokens (special/control tokens that must not be generated) ──
    std::vector<int32_t> banned_token_ids_;

    // ── Stream helpers ───────────────────────────────────────────────
    cudaStream_t prefill_stream() const;
    cudaStream_t decode_stream() const;
    size_t effective_free_vram() const;

    // ── Init sub-phases ────────────────────────────────────────────
    bool init_weights();
    bool init_kv_cache();
    bool init_features();
    void warmup();

    // ── Inference helpers ────────────────────────────────────────────
    bool is_stop_token(int32_t token) const;

    // Whether the model arch + KV dtype combination supports chunked prefill.
    // Returns true for full-attention models (Qwen3, Llama, Mistral) and
    // hybrid GDN+MoE / Mamba2+MoE models (Qwen3.5/3.6, Nemotron-H) with FP16,
    // FP8, or NVFP4 KV cache. Returns false for SWA archs (Gemma-3/4, Llama-4)
    // and sub-byte KV dtypes lacking gather kernels (INT4, TurboQuant).
    bool supports_chunked_prefill_() const;

    // Resolves config_.prefill_chunk_size considering arch + KV dtype.
    //   sentinel -1 → per-arch default (512 if supported, 0 otherwise)
    //   explicit 0  → 0 (force single-chunk, always respected)
    //   explicit >0 → that value if supported, else 0 with WARN
    int resolve_prefill_chunk_size_() const;

    // Track <think>/<\/think> state and check if generation should stop.
    // Suppresses stop tokens while inside a think block (like llama.cpp).
    void track_think_state(Request& req, int32_t token) const;
    bool should_stop(Request& req, int32_t token) const;

    // Think token IDs (cached from chat template init, -1 if not a think model)
    int32_t think_start_id_ = -1;
    int32_t think_end_id_ = -1;
    void upload_penalties(const Request& req, InferenceState& state, cudaStream_t stream);
    void fill_sampling_params(const Request& req, InferenceState& state) const;
    void fill_recurrent_state(const Request& req, InferenceState& state, bool reset, cudaStream_t stream);
    void finish_request(std::shared_ptr<Request>& req);

    // ── step() sub-phases ─────────────────────────────────────────────
    // Returns: 0 = no async graph active, 1 = still running (step returns true),
    //         -1 = graph exhausted/generation done (check scheduler for more work)
    int step_async_graph_resume();

    // Schedule prefill/decode batches and reconfigure green contexts.
    // Returns true if there is work to do (batches non-empty).
    bool step_schedule();

    // Process all prefill requests in sched_prefill_batch_.
    void step_prefill(cudaStream_t stream);

    // Process one prefill request (called from step_prefill).
    void step_prefill_one(std::shared_ptr<Request>& req, int effective_chunk, cudaStream_t stream);

    // Process all decode requests in sched_decode_batch_.
    void step_decode(cudaStream_t stream);

    // Build batched decode state, run forward pass, sample tokens.
    void step_decode_forward(std::vector<std::shared_ptr<Request>>& valid_decode, cudaStream_t stream);

    // Extract logprobs from decode logits and distribute tokens to requests.
    void step_decode_process_outputs(std::vector<std::shared_ptr<Request>>& valid_decode,
                                     const std::vector<int32_t>& tokens, const Tensor& decode_logits_out,
                                     bool needs_logprobs, bool needs_json_mode, bool needs_schema_mode,
                                     cudaStream_t stream);

    // ── CUDA graph helpers ───────────────────────────────────────────
    // Pre-allocate KV blocks and check preconditions for graph loop.
    // Returns remaining tokens (>0 = ok, <=0 = cannot use graph).
    int prepare_graph_loop(std::shared_ptr<Request>& req);

    // Build InferenceState + CudaGraphConditionalRunner::Config for graph loop.
    CudaGraphConditionalRunner::Config build_graph_config(const Request& req, int remaining) const;

    std::vector<int32_t> try_graph_loop_decode(std::shared_ptr<Request> req, int32_t first_token,
                                               cudaStream_t stream);
    bool try_launch_async_graph_loop(std::shared_ptr<Request> req, int32_t first_token, cudaStream_t stream);
};

}  // namespace imp
