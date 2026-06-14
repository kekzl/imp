#pragma once

#include "model/model.h"
#include "model/chat_template.h"
#include "runtime/scheduler.h"
#include "runtime/request.h"
#include "runtime/batch.h"
#include "runtime/green_ctx.h"
#include "runtime/cuda_graph.h"
#include "vision/vision_pipeline.h"
#include "runtime/constraint_manager.h"
#include "runtime/config.h"
#include "memory/kv_cache.h"
#include "memory/kv_cache_manager.h"
#include "memory/ssm_state.h"
#include "memory/gdn_state.h"
#include "memory/layer_offload.h"
#include "memory/memory_manager.h"
#include "memory/vram_allocator.h"
#include "exec/executor.h"
#include "core/cuda_raii.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>

#include "lora/lora_adapter.h"

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

    // [DEPRECATED] TurboQuant sketch multiplier — retained for ABI compat, not used.
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
    int prefill_chunk_size = -1;  // -1 = per-arch default (2048 if supported, 0 otherwise)

    // FP8 prefill weight cache: uses FP8 E4M3 instead of FP16 for ~2x prefill throughput
    bool use_fp8_prefill = false;

    // NVFP4 decode weight cache: -1=auto, 0=off, 1=additive (FP16+NVFP4), 2=NVFP4 only
    int use_nvfp4_decode = -1;
    bool nvfp4_decode_all = false;  // extend NVFP4 decode cache to Q4_K/Q3_K/Q2_K

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
    // Cap on cache_control/cache_prompt-pinned blocks, percent of the KV
    // pool (0 = pin requests are ignored).
    int prefix_pin_budget_pct = 25;

    // Vision (multimodal)
    std::string mmproj_path;  // path to mmproj GGUF, empty = text-only

    // StreamingLLM smart KV cache (Xiao et al., 2023): keep first N "sink"
    // tokens + last W tokens, drop everything in between. Reduces decode KV
    // bandwidth and enables long-running generations without VRAM blowup.
    // Currently active only on the FP16 GQA decode path; quantized variants
    // ignore these settings.
    bool streaming_kv_enabled = false;
    bool streaming_kv_auto = true;   // auto-enable StreamingLLM when KV cache >90% full
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

    // ---- LoRA hot-swap (issue #522) ----
    // Adapters are activation-path low-rank deltas (no weight patching), so
    // they compose with every quant tier. Swapping invalidates decode graphs
    // (captures hold the adapter's kernels/pointers); swap between requests.
    // Returns adapter id >= 1, or 0 on failure. id 0 = base model.
    int lora_load(const std::string& path);
    bool lora_set(int id);  // 0 deactivates
    int lora_active() const { return active_lora_; }

    // Reset batch pool upload cache (call on context_reset to prevent
    // stale block table pointers when KV blocks are reused)
    void reset_batch_pool_cache();

    // Chunked-prefill-aware teacher-forced perplexity (imp_perplexity).
    // begin: upload the target tokens + zero a per-position NLL buffer.
    // While active, step_prefill_one accumulates -log p(next token) for every
    // chunk it forwards (the executor's hidden_ only retains the most recent
    // chunk, so a post-hoc executor()->perplexity_nll() reads stale positions
    // whenever the resolved prefill chunk size is smaller than the corpus).
    // Prefix-cache block reuse is bypassed while active so every position is
    // actually forwarded. end: fixed-order host reduction (bit-reproducible),
    // returns exp(mean NLL) in *out_ppl and always frees the buffers.
    [[nodiscard]] bool begin_perplexity_capture(const int32_t* tokens, int n);
    [[nodiscard]] bool end_perplexity_capture(double* out_ppl);

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

    // Run MTP forward across all prompt positions to populate the MTP-side
    // KV cache before decode starts. Without this, MTP enters decode with an
    // empty KV cache while the main model has the entire prompt context —
    // a fundamental asymmetry that caps achievable accept rate.
    //
    // Inputs:
    //   prompt_tokens : the full prompt token ids (host array of length n)
    //   d_hidden      : device buffer [n, hidden_dim] FP16 — main-model hidden
    //                   states for every prompt position (executor's hidden_
    //                   buffer right after the prefill forward).
    //   n             : number of prompt tokens
    // Side effects:
    //   - Advances ws.mtp_pos from 0 to n.
    //   - Stores the LAST position's MTP prediction in mtp_pending_prediction_
    //     so that the first decode step's accuracy is measured correctly.
    // Returns false if MTP is disabled or any forward fails (best-effort).
    bool mtp_prefill_prompt(const int32_t* prompt_tokens, const void* d_hidden, int n);

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
    // Per-lookahead accept rate (Phase 3.5 multi-step diagnostic).
    // chain_accept_[k] tracks drafts that were the (k+1)-th in a chain at draft
    // time. chain_accept_[0] == mtp_accuracy_ (next-step prediction).
    // chain_accept_[1] is the second draft (predicts 2 steps ahead); etc.
    std::vector<MtpAccuracy> mtp_chain_accept() const noexcept { return mtp_chain_accept_; }
    void mtp_accuracy_reset() noexcept;  // also resets MTP KV cache pos

    // Accessors for C API
    Scheduler* scheduler() const noexcept { return scheduler_.get(); }
    KVCacheManager* kv_manager() const noexcept { return kv_manager_.get(); }
    KVCache* kv_cache() const noexcept { return kv_cache_raw_; }
    Model* model() const noexcept { return model_.get(); }
    // Effective context window actually allocated by the engine (after VRAM-aware
    // auto-sizing in init_compute_max_seq_len_). May be < the model's declared
    // max context when VRAM is tight — callers MUST gate prompt length on this,
    // not on model().config().max_seq_len, or an over-long prompt overruns the
    // KV/position buffers (SIGSEGV instead of a clean rejection).
    int max_seq_len() const noexcept { return config_.max_seq_len; }
    const ChatTemplate& chat_template() const noexcept { return chat_template_; }
    const std::vector<int32_t>& banned_token_ids() const { return banned_token_ids_; }
    GraphExecutor* executor() const noexcept { return executor_.get(); }
    VRAMAllocator& vram_allocator() noexcept { return memory_manager_.vram_allocator(); }
    MemoryManager& memory_manager() noexcept { return memory_manager_; }
    const MemoryManager& memory_manager() const noexcept { return memory_manager_; }

    // Phase 5 Track D: per-Engine RuntimeConfig (replaces RuntimeConfig::current()
    // singleton). Engine::init snapshots the loaded config; engine_init_resolver
    // mutates it in place for arch-specific defaults; GraphExecutor reads a
    // non-owning pointer set via set_runtime_config().
    const RuntimeConfig& runtime_config() const noexcept { return runtime_config_; }
    RuntimeConfig& mutable_runtime_config() noexcept { return runtime_config_; }

private:
    // ── Core components ──────────────────────────────────────────────
    // Phase 5 Track C: façade over VRAMAllocator + (lazy) PinnedAllocator/
    // DeviceAllocator + vram_budget/storage_planner free functions.
    MemoryManager memory_manager_;
    // Phase 5 Track D: per-Engine runtime configuration (replaces the
    // RuntimeConfig::current() process-wide singleton). Snapshot is
    // initialized from take_pending_runtime_config() at the start of
    // Engine::init(). Tool mains (imp-cli, imp-server) stash the loaded
    // RuntimeConfig via set_pending_runtime_config() before constructing
    // the Engine; library/test embeddings without a pending config get
    // a freshly env-seeded default. The snapshot is then mutated in-place
    // by engine_init_resolver helpers for arch-specific defaults
    // (deterministic_gemm, prefill_graph, etc.).
    RuntimeConfig runtime_config_;
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
    //
    // Pool size 64 (was 32 — P5 §2.2 M4): the imp-server's continuous-batching
    // path can dispatch up to config.max_batch_size sequences per decode step.
    // Default max_batch_size is 64 in the server profile; with the old pool of
    // 32 entries any batch beyond 32 sequences fell off the captured fast path
    // into eager forward, costing ~10× per-step latency. Raising the cap to 64
    // covers the default deployment without extra VRAM (a CudaGraphRunner is a
    // few empty pointers until first capture).
    static constexpr int kMaxGraphPoolSize = 64;
    CudaGraphRunner decode_graph_pool_[kMaxGraphPoolSize];  // index = n_sequences - 1
    std::vector<std::unique_ptr<LoraAdapter>> lora_adapters_;
    int active_lora_ = 0;
    int last_decode_max_blocks_per_graph_[kMaxGraphPoolSize] = {};

    // Prefill graph runner — captures forward_logits for non-last chunks of
    // chunked prefill. Single runner: in practice chunk_len == prefill_chunk_size
    // for all non-last chunks, so per-shape variability collapses to one shape.
    // Opt-in via IMP_PREFILL_GRAPH=1 (Phase 4 of MoE-prefill-graphs work).
    CudaGraphRunner prefill_graph_runner_;
    int last_prefill_chunk_len_ = -1;
    int last_prefill_block_count_ = -1;
    int32_t* h_sample_pinned_ = nullptr;
    // Async conditional graph loop
    CudaGraphConditionalRunner async_graph_runner_;
    std::shared_ptr<Request> async_graph_req_;
    int* async_d_block_tables_ = nullptr;
    int32_t* async_d_banned_tokens_ = nullptr;
    std::vector<int32_t> async_pending_tokens_;
    int async_pending_cursor_ = 0;
    // Burst-hybrid speculation: after a bounded (step_limit) burst drains,
    // the runner stays "parked" with its captured graph + block-table buffer
    // so the next burst of the SAME request can rearm instead of recapture.
    int async_parked_req_id_ = -1;
    int async_bt_capacity_ = 0;  // block-table slots baked into the graph

    // Pipelined constrained decode (json_mode / json_schema, single sequence).
    // Constrained requests can't run the conditional graph loop (the grammar
    // FSM lives on the host), but the eager path leaves the GPU idle during
    // every host turnaround (FSM update + mask compute + relaunch). This mode
    // keeps the host FSM authoritative and hides its latency instead: each
    // tick enqueues [mask + sample + advance] for the in-flight forward AND
    // the NEXT forward (which reads the freshly sampled token from device
    // memory), so the GPU is already deep in forward N+1 while the host
    // processes token N.
    struct ConstrainedPipeline {
        bool active = false;
        std::shared_ptr<Request> req;
        CudaGraphRunner runner;  // captures forward_logits only
        InferenceState state{};  // stable device pointers for the whole run
        Tensor logits;           // fixed-address logits view (workspace buffer)
        int32_t* d_token = nullptr;  // ARGMAX_SCRATCH_BYTES (token + argmax scratch)
        int* d_pos = nullptr;        // [1] current position
        int* d_ctx = nullptr;        // [1] current context length
        int* d_bt = nullptr;         // uploaded block table
        int32_t* d_banned = nullptr; // banned token ids (device)
        int32_t* h_token = nullptr;  // pinned landing for the sampled token
        cudaEvent_t ev = nullptr;    // sampled-token-ready event
        int budget = 0;              // tokens coverable by pre-allocated KV
        int produced = 0;            // tokens harvested by this pipeline
        bool forward_in_flight = false;
    };
    ConstrainedPipeline cpipe_;

    // Teacher-forced perplexity capture (begin/end_perplexity_capture).
    // Device buffers of length n; step_prefill_one writes per-position NLL
    // for every forwarded chunk while active.
    struct PplCapture {
        bool active = false;
        int n = 0;
        int32_t* d_tokens = nullptr;
        double* d_nll = nullptr;
    } ppl_capture_;

    // ── Model-specific state ─────────────────────────────────────────
    std::unique_ptr<SSMState> ssm_state_;
    std::unique_ptr<GDNState> gdn_state_;

    // Recurrent (SSM/GDN) state-slot allocator. One slot per concurrent
    // sequence (capacity == config.max_batch_size). Slots MUST be unique among
    // live sequences — the recurrent state IS the sequence's memory, so a shared
    // slot leaks one request's context into another. The previous
    // `req.id % capacity` scheme aliased slots whenever two live request ids
    // differed by a multiple of capacity; allocate a distinct free slot instead.
    std::vector<int> free_recurrent_slots_;           // available slot ids
    std::unordered_map<int, int> recurrent_slot_of_;  // req.id -> slot
    bool recurrent_slots_initialized_ = false;
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
    // Guards reuse of the pinned staging above: records after the H2D copies
    // are enqueued; the next chunk host-waits on it before REWRITING the
    // pinned source. Without this the host runs many fully-async chunks
    // ahead (no implicit syncs on the FA2 path) and overwrites the staging
    // while earlier copies are still queued -> chunk c uploads chunk c+N's
    // tokens/positions (#548: catastrophic chunked-prefill NLL on Llama).
    cudaEvent_t pf_staging_evt_ = nullptr;

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
    // Defined in <compute/mtp_forward.h>; forward-declared to avoid include.
    int mtp_spec_k_ = 0;
    void* mtp_ws_storage_ = nullptr;  // type-erased MtpDraftWorkspace*

    // Phase 3.5 telemetry: rolling MTP-draft-accuracy across the active session.
    // mtp_pending_prediction_ is the prediction made at the end of the
    // PREVIOUS decode step; it gets compared to the actual next_token at the
    // start of the CURRENT step. -1 = no pending prediction (start of session,
    // batch>1, prediction call failed, etc).
    int mtp_pending_prediction_ = -1;
    MtpAccuracy mtp_accuracy_{};
    // K>1 chain measurement: window of pending predictions. Each entry is
    // (prediction, lookahead_at_draft, intended_position). When the engine
    // generates a token at intended_position, the matching entry is verified
    // and chain_accept_[lookahead] is incremented.
    struct MtpChainEntry {
        int prediction;
        int lookahead;
        int intended_position;
    };
    std::vector<MtpChainEntry> mtp_pending_chain_;
    std::vector<MtpAccuracy> mtp_chain_accept_;

    // ── n-gram (prompt-lookup) speculative decoding ─────────────────
    // Drafts come from suffix matches against the request's own context
    // (runtime_config_.speculative); the verify step replays them as a
    // teacher-forced continuation chunk and accepts the longest greedy-
    // matching prefix. Implemented in engine_spec_ngram.cpp.
    // Returns true when it handled this decode step (tokens emitted);
    // false → caller falls through to the normal decode path.
    bool step_spec_verify_(std::shared_ptr<Request>& req, cudaStream_t stream);
    bool spec_ngram_gates_ok_(const Request& req, bool ignore_think = false) const;
    bool spec_burst_launch_ok_(const Request& req) const;
    int spec_effective_miss_burst_(const Request& req) const;
    void spec_maybe_rearm_(Request& req) const;
    bool ensure_spec_buffers_(int chunk_cap, int max_blocks);
    void free_spec_buffers_();
    void log_spec_stats_() const;
    // Device/pinned staging for the verify chunk (lazy-init, K+1 capacity).
    int32_t* d_spec_tokens_ = nullptr;
    int* d_spec_positions_ = nullptr;
    int* d_spec_block_table_ = nullptr;
    int* d_spec_context_len_ = nullptr;
    int32_t* d_spec_argmax_ = nullptr;
    int32_t* h_spec_argmax_ = nullptr;  // pinned, chunk_cap entries
    int spec_chunk_cap_ = 0;
    int spec_block_table_cap_ = 0;
    // Session telemetry (logged when a request finishes).
    struct SpecStats {
        long long verify_steps = 0;  // verify forwards run
        long long miss_steps = 0;    // decode steps with no usable draft
        long long drafted = 0;       // draft tokens proposed
        long long accepted = 0;      // draft tokens accepted
        long long emitted = 0;       // tokens emitted by verify steps
    };
    SpecStats spec_stats_{};

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
    // Config-resolution helpers for the front half of init() — each one
    // mutates config_ / RuntimeConfig in place, then init() executes
    // init_weights / init_kv_cache / init_features. Order matters: see the
    // call sequence in init() for which flag each one resolves.
    void init_apply_debug_raw_overrides_();
    void init_resolve_kv_dtype_policy_();
    void init_resolve_ssm_dtype_();
    void init_resolve_fp8_prefill_();
    void init_resolve_quant_flags_();
    void init_compute_max_seq_len_();
    // step_prefill_one sub-phase: allocate KV blocks for `req`. Handles
    // prefix-cache reuse + eviction fallback. Returns false on
    // unrecoverable allocation failure (caller cancels request). On
    // successful prefix-cache reuse, advances `offset` / `chunk_len` /
    // `is_last_chunk` / `ctx_len` to skip the cached prefix.
    [[nodiscard]] bool prefill_allocate_kv_blocks_(std::shared_ptr<Request>& req, int kv_bs,
                                                   int total_input, int effective_chunk,
                                                   int& offset, int& chunk_len, bool& is_last_chunk,
                                                   int& ctx_len, cudaStream_t pf_stream);
    // step_prefill_one sub-phase: upload token_ids / positions / block_tables /
    // context_lens to device. Uses the prefill_pool_ buffers when chunk_len <=
    // max_seq_len, otherwise falls back to cudaMallocAsync. Returns false on
    // alloc / memcpy failure (caller sets req->status = CANCELLED, frees its
    // KV blocks). Outputs: device buffer pointers + `pf_pool_used` flag for
    // the matching free path.
    [[nodiscard]] bool prefill_upload_metadata_(std::shared_ptr<Request>& req,
                                                const std::vector<int>& block_table,
                                                int chunk_len, int offset, int ctx_len,
                                                cudaStream_t pf_stream,
                                                int32_t*& d_token_ids, int*& d_positions,
                                                int*& d_block_tables, int*& d_context_lens,
                                                bool& pf_pool_used);
    // step_decode_forward sub-phase: populate `state` from the uploaded
    // GPU batch + per-seq residual metadata + sampling params + recurrent
    // state + JSON/schema constrainers. Returns `needs_logprobs` so the
    // caller knows whether to capture decode_logits_out for the logprobs
    // pass downstream.
    void decode_build_inference_state_(GPUBatch& gpu_batch,
                                       std::vector<std::shared_ptr<imp::Request>>& valid_decode,
                                       int max_ctx, cudaStream_t dec_stream,
                                       InferenceState& state, bool& needs_logprobs,
                                       bool& needs_json_mode, bool& needs_schema_mode);

    // Build banned_token_ids_ — special/control tokens that must never appear
    // in generated output (e.g. <|im_start|>, <|endoftext|>). Scans tokenizer
    // for control-tagged tokens (authoritative GGUF token_types) or falls back
    // to heuristic patterns. Excludes stop/EOS/think/channel tokens. Bypassed
    // by IMP_NO_BAN=1 for bisecting NVFP4 repetition issues.
    void build_banned_token_list();

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
    int acquire_recurrent_slot_(int req_id);   // distinct free slot for a new sequence
    void release_recurrent_slot_(int req_id);  // idempotent; returns slot to the pool
    void finish_request(std::shared_ptr<Request>& req);

    // ── step() sub-phases ─────────────────────────────────────────────
    // Returns: 0 = no async graph active, 1 = still running (step returns true),
    //         -1 = graph exhausted/generation done (check scheduler for more work)
    int step_async_graph_resume();

    // Pipelined constrained decode (see ConstrainedPipeline). Launch after the
    // first decode step of an eligible json/schema request; one tick harvests
    // one token. Returns: 0 = inactive/exhausted (fall through to eager),
    // 1 = produced a token and continues, -1 = generation finished.
    bool try_launch_constrained_pipeline(std::shared_ptr<Request> req, cudaStream_t stream);
    int step_constrained_pipeline();
    void teardown_constrained_pipeline(bool synchronize);

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
    bool try_launch_async_graph_loop(std::shared_ptr<Request> req, int32_t first_token,
                                     cudaStream_t stream, int step_limit = 0);
};

}  // namespace imp
