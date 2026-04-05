#pragma once

#include "model/model.h"
#include "model/chat_template.h"
#include "runtime/scheduler.h"
#include "runtime/request.h"
#include "runtime/batch.h"
#include "runtime/green_ctx.h"
#include "runtime/cuda_graph.h"
#include "runtime/speculative.h"
#include "runtime/self_speculative.h"
#include "runtime/ngram_spec.h"
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
    int max_batch_size = 32;
    int max_seq_len = 4096;
    int kv_cache_max_blocks = 0;  // 0 = auto
    bool use_green_contexts = false;
    float green_ctx_prefill_ratio = 0.8f;
    bool use_cuda_graphs = true;
    bool use_pdl = true;
    DType compute_dtype = DType::FP16;

    // Default sampling parameters
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = -1;

    // KV cache dtype: FP16 (default) or FP8_E4M3 for ~50% KV VRAM savings
    DType kv_cache_dtype = DType::FP16;

    // TurboQuant Lite: sketch_dim = turboquant_sketch_multiplier * head_dim
    int turboquant_sketch_multiplier = 2;

    // SSM state dtype: FP32 (default) or FP16 for ~50% VRAM savings on h_state
    DType ssm_state_dtype = DType::FP32;

    // VRAM budget: max GPU memory to use (MiB), 0 = use all available
    size_t vram_budget_mb = 0;

    // Layer offloading: number of layers to keep on GPU (-1 = all on GPU, 0 = all offloaded)
    int gpu_layers = -1;

    // KV cache block size (tokens per block). 0 = auto-select based on model.
    // Larger blocks (32, 64) improve coalescing for GQA models with few KV heads.
    int kv_block_size = 0;

    // Chunked prefill
    int prefill_chunk_size = 0;  // 0 = no chunking

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

    // Speculative decoding
    bool enable_speculative = false;
    std::string draft_model_path;
    int spec_k = 4;

    // Prefix caching: reuse KV cache blocks for shared token prefixes
    bool use_prefix_caching = false;
    std::string prefix_cache_path;  // path to persist prefix cache (empty = disabled)

    // Self-speculative decoding (layer-skip draft from same model)
    bool enable_self_speculative = false;
    int self_spec_k = 2;              // draft tokens per step
    int self_spec_exit_layer = -1;    // layers to run in draft (-1 = auto)
    int self_spec_skip_n = -1;        // layers to skip in draft (-1 = auto)

    // N-gram speculative decoding (zero-cost draft from token history)
    bool enable_ngram_spec = false;  // experimental, disabled by default
    int ngram_spec_k = 5;          // max draft tokens per step
    int ngram_n = 3;               // n-gram context window

    // Vision (multimodal)
    std::string mmproj_path;  // path to mmproj GGUF, empty = text-only
};

class Engine {
public:
    Engine() = default;
    ~Engine();

    [[nodiscard]] bool init(std::shared_ptr<Model> model, const EngineConfig& config);

    // Run one step of inference (prefill or decode depending on scheduler)
    [[nodiscard]] bool step();

    // High-level generate with sampling parameters
    std::string generate(const std::string& prompt, int max_tokens,
                         float temperature = 1.0f, float top_p = 1.0f,
                         int top_k = 0, int seed = -1,
                         bool apply_chat_template = true,
                         float min_p = 0.0f,
                         float repetition_penalty = 1.0f,
                         float frequency_penalty = 0.0f,
                         float presence_penalty = 0.0f);

    void add_request(std::shared_ptr<Request> req);

    // Set draft model for speculative decoding after init.
    // Can only be called once, before any generate/decode_step calls.
    [[nodiscard]] bool set_draft_model(const std::string& path, int spec_k = 4);

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
    std::unique_ptr<LayerOffloadManager> offload_mgr_;
    bool experts_on_host_ = false;
    bool dequant_done_ = false;
    ChatTemplate chat_template_;

    // ── Extracted subsystems ─────────────────────────────────────────
    VisionPipeline vision_;
    ConstraintManager constraints_;

    // ── Speculative decoding ─────────────────────────────────────────
    std::shared_ptr<Model> draft_model_;
    std::unique_ptr<KVCacheManager> draft_kv_manager_;
    std::unique_ptr<SpeculativeDecoder> spec_decoder_;
    std::unique_ptr<SelfSpeculativeDecoder> self_spec_decoder_;
    std::unique_ptr<NgramSpecDecoder> ngram_spec_decoder_;

    // ── Pre-allocated prefill metadata (eliminates per-request cudaMalloc) ──
    void* prefill_pool_ = nullptr;
    size_t prefill_pool_size_ = 0;
    int32_t* d_pf_token_ids_ = nullptr;
    int* d_pf_positions_ = nullptr;
    int* d_pf_block_tables_ = nullptr;
    int* d_pf_context_lens_ = nullptr;
    int* h_pf_positions_ = nullptr;       // pinned host staging
    int32_t* h_pf_token_ids_ = nullptr;   // pinned host staging

    // ── Penalty token buffer ─────────────────────────────────────────
    int32_t* d_penalty_tokens_ = nullptr;
    size_t d_penalty_tokens_capacity_ = 0;

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
    bool init_speculative();

    // ── Inference helpers ────────────────────────────────────────────
    bool is_stop_token(int32_t token) const;

    // Track <think>/<\/think> state and check if generation should stop.
    // Suppresses stop tokens while inside a think block (like llama.cpp).
    void track_think_state(Request& req, int32_t token) const;
    bool should_stop(Request& req, int32_t token) const;

    // Think token IDs (cached from chat template init, -1 if not a think model)
    int32_t think_start_id_ = -1;
    int32_t think_end_id_ = -1;
    void upload_penalties(const Request& req, InferenceState& state,
                          cudaStream_t stream);
    void fill_sampling_params(const Request& req, InferenceState& state) const;
    void fill_recurrent_state(const Request& req, InferenceState& state,
                               bool reset, cudaStream_t stream);
    void finish_request(std::shared_ptr<Request>& req);

    // Speculative decode shortcuts (self-spec, n-gram). Returns true if handled.
    bool try_speculative_decode(std::vector<std::shared_ptr<Request>>& valid_decode,
                                 cudaStream_t stream);

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
    void step_prefill_one(std::shared_ptr<Request>& req, int effective_chunk,
                          cudaStream_t stream);

    // Process all decode requests in sched_decode_batch_.
    void step_decode(cudaStream_t stream);

    // Build batched decode state, run forward pass, sample tokens.
    void step_decode_forward(std::vector<std::shared_ptr<Request>>& valid_decode,
                             cudaStream_t stream);

    // Extract logprobs from decode logits and distribute tokens to requests.
    void step_decode_process_outputs(std::vector<std::shared_ptr<Request>>& valid_decode,
                                     const std::vector<int32_t>& tokens,
                                     const Tensor& decode_logits_out,
                                     bool needs_logprobs, bool needs_json_mode,
                                     bool needs_schema_mode, cudaStream_t stream);

    // ── CUDA graph helpers ───────────────────────────────────────────
    // Pre-allocate KV blocks and check preconditions for graph loop.
    // Returns remaining tokens (>0 = ok, <=0 = cannot use graph).
    int prepare_graph_loop(std::shared_ptr<Request>& req);

    // Build InferenceState + CudaGraphConditionalRunner::Config for graph loop.
    CudaGraphConditionalRunner::Config build_graph_config(
        const Request& req, int remaining) const;

    std::vector<int32_t> try_graph_loop_decode(
        std::shared_ptr<Request> req, int32_t first_token, cudaStream_t stream);
    bool try_launch_async_graph_loop(std::shared_ptr<Request> req,
                                     int32_t first_token, cudaStream_t stream);
};

} // namespace imp
