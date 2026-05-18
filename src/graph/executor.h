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
#include "graph/weight_handle.h"
#include "graph/moe_workspace.h"
#include "graph/quant_scratch.h"
#include "runtime/storage_planner.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <unordered_map>
#include <utility>
#include <list>

namespace imp {

// ---------------------------------------------------------------------------
// LRU cache for GPU-resident expert weights.
// When MoE experts don't fit in VRAM, they reside on host (mmap/pinned).
// This cache keeps recently-used experts on GPU to avoid repeated H2D copies.
// Key = (packed_tensor_ptr, expert_index), Value = GPU slot with raw bytes.
// ---------------------------------------------------------------------------
struct ExpertCacheKey {
    const void* packed_ptr;  // pointer to packed tensor (identifies weight matrix)
    int expert_idx;
    bool operator==(const ExpertCacheKey& o) const {
        return packed_ptr == o.packed_ptr && expert_idx == o.expert_idx;
    }
};

struct ExpertCacheKeyHash {
    size_t operator()(const ExpertCacheKey& k) const {
        // Combine pointer hash and expert index
        size_t h = std::hash<const void*>{}(k.packed_ptr);
        h ^= std::hash<int>{}(k.expert_idx) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// MoE projection enum — every layer has up to three: gate, up, down.
// Used to index the per-layer × per-projection × per-expert device-side
// lookup mirror (Phase 2 of the MoE host-offload + CUDA Graphs design,
// see docs/plans/moe_host_offload_graphs_design_2026_05_17.md §3a).
enum class ExpertProj { Gate = 0, Up = 1, Down = 2 };
inline constexpr int kExpertProjCount = 3;

// Per-layer LRU state — Phase 3 partitions the global pool so layer L's
// cache state can't be evicted by layer M's misses. Each layer has its own
// recency list + key→slot map; slot indices are layer-relative.
struct PerLayerLRU {
    std::list<int> lru_order;  // slot indices WITHIN this layer (0..slots_per_layer-1)
    using LRUIter = std::list<int>::iterator;
    std::unordered_map<ExpertCacheKey, std::pair<int, LRUIter>, ExpertCacheKeyHash> lookup;
};

// Per-layer access history ring — Phase 4 prefetch signal. Records every
// get_or_load(layer, proj, expert) hit OR miss into a fixed-size ring, so
// `prefetch_layer(layer)` can consult the most-recent (proj, expert) pairs
// even after they've been evicted from the LRU. Recency-resilient — survives
// across-token churn that the per-layer LRU's "currently cached" view doesn't.
struct PerLayerAccessRing {
    // Capacity is per-layer; sized at init so writes never reallocate.
    std::vector<std::pair<int, int>> entries;  // (proj, expert) pairs
    int head = 0;       // next write position (mod entries.size())
    int filled = 0;     // entries actually populated (clamped to capacity)
};

struct ExpertLRUCache {
    // Each slot holds one expert's raw quantized bytes on GPU. Slots are
    // partitioned per-layer (Phase 3): `slots_[layer * slots_per_layer_ +
    // slot_idx_within_layer]`. GPU pointer = `pool_ + flat_index * slot_size_`.
    struct Slot {
        void* gpu_ptr = nullptr;  // points into pool_
        ExpertCacheKey key = {};
        bool occupied = false;
        // Mirror coords — set on get_or_load, used to invalidate the device
        // lookup cell on eviction. `layer` matches the flat-index layer.
        int layer = -1;
        int proj = -1;   // ExpertProj
        int expert = -1;
    };

    void* pool_ = nullptr;     // contiguous GPU allocation for all slots
    size_t slot_size_ = 0;     // bytes per slot (max expert raw size)
    int slots_per_layer_ = 0;  // S — uniform per-layer pool depth (Phase 3)
    int n_slots_ = 0;          // = n_layers_ * slots_per_layer_ (total)
    std::vector<Slot> slots_;  // flat array, layer-major (size = n_slots_)

    // Per-layer LRU state (Phase 3). `per_layer_lru_[layer]` holds layer L's
    // recency list + key→slot map. Slot indices are layer-relative (0..S-1).
    std::vector<PerLayerLRU> per_layer_lru_;
    using LRUIter = PerLayerLRU::LRUIter;

    // Phase 4: per-layer access history ring + prefetch infrastructure.
    // - `per_layer_history_[layer]` records every get_or_load() (proj, expert)
    //   into a fixed-size ring. Even after eviction the access is remembered,
    //   so prefetch_layer() can pre-warm the cache with experts likely to
    //   recur next token.
    // - `prefetch_stream_` is a dedicated CUDA stream where prefetch H2Ds run
    //   concurrent with the engine's compute stream. Compute waits on
    //   `prefetch_done_[layer]` before dispatching layer L's reads.
    std::vector<PerLayerAccessRing> per_layer_history_;
    int history_capacity_ = 0;        // entries per layer (sized at init)
    cudaStream_t prefetch_stream_ = nullptr;
    std::vector<cudaEvent_t> prefetch_done_;  // one event per layer, signaled by prefetch_layer
    std::vector<bool> prefetch_issued_;       // per-layer flag — has prefetch_layer been called?
    int64_t prefetch_h2ds_ = 0;       // count of async H2Ds the prefetcher actually issued
    int64_t prefetch_skipped_cached_ = 0;  // ring entries already cached at prefetch time

    // Phase 3: pre-computed host source pointers for capture-safe memcpy.
    // host_expert_addrs_[layer][proj * n_experts + expert] = packed.data +
    // expert_idx * expert_raw. Populated lazily on first get_or_load per
    // cell — the value is stable for the lifetime of the model load (packed
    // tensors don't move on host). Phase 5 will use these as the fixed
    // `src` argument of cudaGraphAddMemcpyNode so the graph can replay
    // without recomputing host pointers each iteration.
    std::vector<std::vector<const void*>> host_expert_addrs_;

    // Phase 4: canonical packed_ptr per (layer, proj). The ExpertCacheKey
    // hashes on packed_ptr, so prefetch must rebuild keys that match what
    // dispatch will pass on its next get_or_load. Sized [n_layers × 3];
    // populated on first get_or_load per (layer, proj).
    std::vector<const void*> host_packed_ptrs_;

    // Phase 5: per-(layer, proj) expert byte size. The prefetch path can't
    // reuse the global slot_size_ (max across projs) because smaller
    // projections — e.g. Qwen3.6 gate @ 144 MiB vs down @ 176 MiB — would
    // overflow the pinned host region during cudaMemcpyAsync and fail with
    // "invalid argument". Populated on first get_or_load per (layer, proj)
    // when the caller passes the actual per-expert byte size.
    std::vector<size_t> host_expert_bytes_;

    int64_t hits_ = 0;
    int64_t misses_ = 0;

    VRAMAllocator* alloc_ = nullptr;

    // Device-side lookup mirror (Phase 2 → Phase 3). Sized [n_layers × 3 ×
    // n_experts] int32 cells; cell value = **layer-relative** slot_idx
    // (0..slots_per_layer-1) or -1 if not cached. Read pattern (Phase 5 will
    // do this from inside dispatch kernels):
    //   slot = d_lookup_[layer * 3 * n_experts + proj * n_experts + expert];
    //   if (slot >= 0) src = pool_ + (layer * slots_per_layer + slot) * slot_size;
    int* d_lookup_ = nullptr;
    int n_layers_ = 0;
    int n_experts_ = 0;
    bool debug_parity_ = false;
    mutable int64_t parity_checks_ok_ = 0;  // exposed for tests; bumped by const check_parity()

    // Initialize: allocate the slot pool (partitioned per-layer) + the
    // device mirror + the host source-pointer table. Returns false if GPU
    // allocation fails (cache disabled).
    bool init(size_t max_expert_raw, size_t budget_bytes, VRAMAllocator* alloc,
              int n_layers, int n_experts, bool debug_parity = false);

    // Lookup or insert an expert. Returns GPU pointer to cached expert data.
    // If cache miss: copies from host, evicts LRU entry within the layer's
    // pool if needed.
    // src_host = host pointer to this expert's raw bytes. The first call per
    // (layer, proj, expert) records src_host into host_expert_addrs_;
    // subsequent calls reuse the stored pointer (the cache key disambiguates).
    void* get_or_load(int layer, ExpertProj proj, ExpertCacheKey key,
                      const void* src_host, size_t expert_bytes, cudaStream_t stream);

    // Check if (layer, expert) is cached (no insertion). Updates LRU recency
    // within the layer.
    void* find(int layer, ExpertCacheKey key);

    void destroy();

    float hit_rate() const {
        int64_t total = hits_ + misses_;
        return total > 0 ? static_cast<float>(hits_) / total : 0.0f;
    }

    // Returns the device pointer for layer L, slot_idx_within_layer s, or
    // nullptr if the slot is not occupied. Used by Phase 3 tests and as the
    // future Phase 5 kernel-side resolve helper. Phase 5 will inline this
    // logic into the dispatch kernels.
    void* slot_ptr(int layer, int slot_idx_within_layer) const {
        if (!pool_ || slot_idx_within_layer < 0 ||
            slot_idx_within_layer >= slots_per_layer_)
            return nullptr;
        size_t flat = static_cast<size_t>(layer) * slots_per_layer_ + slot_idx_within_layer;
        return static_cast<char*>(pool_) + flat * slot_size_;
    }

    // Phase 4 — async prefetch.
    //
    // prefetch_layer(layer, top_k, expert_bytes): walk per_layer_history_[layer]
    // from the most-recent entry backward, gather up to top_k unique
    // (proj, expert) pairs that AREN'T currently cached, and kick off an
    // async H2D into a freshly-allocated slot for each on prefetch_stream_.
    // The host LRU + device mirror are updated synchronously on the host
    // side, so a follow-up get_or_load(layer, …) on the compute stream sees
    // the slot as "cached". Records prefetch_done_[layer] when the H2Ds
    // finish — the compute stream must wait on this before reading any
    // affected slot.
    //
    // expert_bytes is the H2D copy size; pass `slot_size_` if you don't
    // know it precisely (over-copy is wasted bandwidth but correct).
    // Returns the number of async H2Ds issued (0..top_k). No-op if
    // prefetch_stream_ is null or top_k <= 0.
    int prefetch_layer(int layer, int top_k, size_t expert_bytes);

    // Compute stream waits on prefetch_done_[layer]. Safe to call even if
    // prefetch_layer wasn't called for `layer` (skip on no-issue).
    void await_prefetch(int layer, cudaStream_t compute_stream);

    // Phase 2 debug helper: copy the device lookup mirror back to host and
    // assert every cell matches the host-side LRU state. Returns true on
    // match. When `debug_parity_` is enabled, get_or_load() runs this after
    // every cache mutation and bumps `parity_checks_ok_` on success / aborts
    // (via IMP_LOG_FATAL) on mismatch. Tests can call it directly.
    bool check_parity(cudaStream_t stream) const;
};

// VRAM budget for weight cache allocation (computed by Engine::plan_vram_budget).
// Replaces ad-hoc "remaining_budget" with per-phase caps computed upfront.
struct VRAMBudget {
    enum Strategy { FP8_PREFILL_NVFP4_DECODE, NVFP4_DECODE_ONLY, FP16_ONLY };
    Strategy strategy = FP16_ONLY;
    size_t kv_cache_bytes = 0;
    size_t fp8_cache_bytes = 0;  // 0 for sub-8-bit models
    size_t nvfp4_cache_bytes = 0;
    size_t reserve_bytes = 1024ULL * 1024 * 1024;  // 1 GiB safety
    int kv_max_blocks = 0;
    bool nvfp4_second_pass = false;  // true → re-run NVFP4 after FP16-Free
};

// All the state needed for a single forward pass invocation.
struct InferenceState {
    // Input tokens
    const int32_t* token_ids = nullptr;  // [n_tokens] on device
    const int* positions = nullptr;      // [n_tokens] on device
    int n_tokens = 0;

    // KV cache for paged attention (decode)
    KVCache* kv_cache = nullptr;
    const int* block_tables = nullptr;  // [n_sequences, max_blocks_per_seq] on device (2D padded)
    const int* context_lens = nullptr;  // [n_sequences] on device
    int max_context_len = 0;

    // SSM state for Mamba2 layers (nullptr for non-hybrid models)
    SSMState* ssm_state = nullptr;
    int ssm_seq_id = 0;  // sequence ID for SSM state access

    // GDN state for Gated DeltaNet layers (Qwen3.5, nullptr for non-GDN models)
    class GDNState* gdn_state = nullptr;
    int gdn_seq_id = 0;

    // BitDecoding Phase 3 residual KV cache.
    //
    // Two activation modes: single-seq scalar OR multi-seq array. The
    // attention dispatcher prefers the multi-seq form whenever the engine
    // sets up the device arrays (see d_residual_seq_slots below).
    //
    // Single-seq mode (legacy / batch_size==1):
    //   `kv_seq_id` carries the seq_id (= request id) used to look up the
    //   ring state via KVCacheManager. -1 disables.
    int kv_seq_id = -1;
    // KVCacheManager owning the residual buffer + ring-state map.
    class KVCacheManager* kv_manager = nullptr;
    // Multi-seq array form: device pointers to per-batch metadata, all of
    // length n_sequences. Built by the engine on each forward step before
    // the attention call. Each element matches the corresponding row of
    // block_tables / context_lens. nullptr = multi-seq form inactive.
    const int* d_residual_seq_slots = nullptr;     // [n_sequences] slot in [0, residual_max_seqs)
    const int* d_residual_counts = nullptr;         // [n_sequences] fill_count
    const int* d_residual_write_idxes = nullptr;    // [n_sequences] write_idx
    // Host array of per-batch seq_ids (request ids), used by the KV write path
    // to call KVCacheManager::advance_residual per seq. Length = n_sequences.
    const int* h_residual_seq_ids = nullptr;

    // Batching
    int n_sequences = 1;         // number of sequences in the batch
    int max_blocks_per_seq = 0;  // max blocks per sequence (for 2D block_table indexing)
    const int* seq_offsets =
        nullptr;  // [n_sequences+1] for ragged prefill token offsets (optional, nullptr for decode)

    // Mode
    bool is_prefill = true;
    // Absolute position of state.positions[0] within the full sequence.
    // 0 means single-chunk prefill or first chunk of a chunked prefill.
    // > 0 means a follow-up chunk: tokens [0, prefill_offset) are already in the KV cache.
    int prefill_offset = 0;

    // Sampling parameters
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    int seed = -1;
    float min_p = 0.0f;
    float typical_p = 1.0f;  // Locally typical sampling (1.0 = disabled)
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int repeat_last_n = 0;  // How many recent tokens to scan (0 = all)

    // DRY (Don't Repeat Yourself) penalty
    float dry_multiplier = 0.0f;  // 0 = disabled
    float dry_base = 1.75f;
    int dry_allowed_length = 2;
    int dry_penalty_last_n = 0;                    // 0 = full history
    const int32_t* host_penalty_tokens = nullptr;  // HOST pointer for DRY scanning

    // Mirostat v2 adaptive entropy sampling
    int mirostat = 0;                  // 0=off, 2=Mirostat v2
    float mirostat_tau = 5.0f;         // Target entropy
    float mirostat_eta = 0.1f;         // Learning rate
    mutable float mirostat_mu = 0.0f;  // Running variable (updated by sampling)

    // Token history for penalty computation (device pointer, owned by engine)
    const int32_t* penalty_tokens = nullptr;
    int n_penalty_tokens = 0;
    // Device-side penalty token count (for CUDA graph loop where count grows
    // each iteration). When non-null, forward_decode_async reads the count
    // from *d_n_penalty_tokens instead of n_penalty_tokens.
    const int* d_n_penalty_tokens = nullptr;

    // Logprobs: when true, forward() copies logits to h_logits_pinned_ for CPU extraction
    bool logprobs = false;
    int top_logprobs = 0;

    // JSON mode: when non-null, apply logit mask before sampling
    JsonConstrainer* json_constrainer = nullptr;
    SchemaConstrainer* schema_constrainer = nullptr;

    // Logit bias (host-side, applied via cudaMemcpy before sampling)
    const std::pair<int32_t, float>* logit_bias = nullptr;
    int n_logit_bias = 0;

    // Banned tokens: set logits to -inf before sampling (e.g. chat template special tokens)
    const int32_t* banned_tokens = nullptr;  // HOST pointer, small list
    int n_banned_tokens = 0;
    const int32_t* d_banned_tokens = nullptr;  // DEVICE pointer (for CUDA graph path)
    int n_d_banned_tokens = 0;

    // Force token: when >= 0, set ALL logits except this token to -inf.
    // Used by think-budget to force </think> generation via logit manipulation
    // so the token lands correctly in the KV cache (NVIDIA NIM approach).
    int32_t force_token = -1;

    // Vision: when non-null, replace vision_token_id positions with vision embeddings
    const half* vision_embeddings = nullptr;  // [n_vision_tokens, d_model] FP16 on device
    int vision_token_id = -1;                 // <image_soft_token> ID
    int n_vision_tokens = 0;                  // 256

    // Early exit: run only the first exit_layer layers (-1 = all layers).
    // Used by self-speculative decoding to generate cheap draft tokens.
    int exit_layer = -1;

    // Layer skip: skip layers in [skip_layer_start, skip_layer_end) during forward.
    // Used by self-speculative decoding for better acceptance than pure early exit.
    // Runs layers {0..skip_start-1, skip_end..n_layers-1}. -1 = disabled.
    int skip_layer_start = -1;
    int skip_layer_end = -1;

    // When true, project ALL tokens through the LM head during prefill
    // (normally only the last token is projected). Used by speculative verify.
    bool all_logits = false;

    // When true, bypass FP8 GEMM paths and use dequant→FP16 GEMM instead.
    // Avoids compound FP8 quantization error over many layers (self-spec verify).
    bool force_fp16_gemm = false;

    // When true, use per-row Q8_1 GEMV for LM head instead of batched FP8 GEMM.
    // Avoids FP8 per-tensor quantization artifacts in batched verification.
    bool per_row_lm_head = false;
};

// ---------------------------------------------------------------------------
// FP8 weight cache entry (used by WeightCaches::fp8).
// ---------------------------------------------------------------------------
struct FP8CacheEntry {
    Tensor weight;     // [N, K] FP8_E4M3 on device
    float host_scale;  // absmax / 448
    float* d_scale;    // device-side scale (1 float)
};

// ---------------------------------------------------------------------------
// WeightCaches: all pre-quantized weight maps for the inference engine.
//
// Replaces the former WeightCacheManager type (Phase 5 cleanup).
// All members are public for zero-overhead access in the forward pass.
// Lifecycle: allocated during pre_dequant_weights(), freed in free_buffers().
// ---------------------------------------------------------------------------
struct WeightCaches {
    // --- FP16 weight cache ---
    std::unordered_map<const void*, Tensor> fp16;
    size_t fp16_bytes = 0;

    // Fused KV: [wk; wv] per layer for strided batched prefill GEMM.
    std::unordered_map<int, Tensor> fused_kv;
    // Fused gate+up: [w_gate; w_up] per layer.
    std::unordered_map<int, Tensor> fused_gate_up;

    // --- FP8 E4M3 weight cache ---
    std::unordered_map<const void*, FP8CacheEntry> fp8;
    size_t fp8_bytes = 0;
    bool use_fp8 = false;

    // Bulk-allocated buffers for FP16→FP8 migration
    float* fp8_migrated_scales = nullptr;
    int fp8_migrated_count = 0;
    void* fp8_migrated_data = nullptr;
    size_t fp8_migrated_data_size = 0;

    // Overflow FP8 cache
    float* fp8_overflow_scales = nullptr;
    int fp8_overflow_count = 0;
    void* fp8_overflow_data = nullptr;
    size_t fp8_overflow_data_size = 0;

    // --- NVFP4 decode weight cache ---
    // Mode: 0=off, 1=additive, 2=only
    std::unordered_map<const void*, NvFP4QuantResult> nvfp4;
    size_t nvfp4_bytes = 0;
    int nvfp4_decode_mode = 0;

    // Per-expert NVFP4
    std::unordered_map<const void*, NvFP4MoEQuantResult> nvfp4_moe;
    size_t nvfp4_moe_bytes = 0;

    // --- CUTLASS sm_120 block-scaled NVFP4 ---
    std::unordered_map<const void*, CutlassNvFP4Weight> cutlass_nvfp4;
    size_t cutlass_nvfp4_bytes = 0;

    // --- CUTLASS sm_120 MXFP4 ---
    std::unordered_map<const void*, CutlassMxFP4Weight> cutlass_mxfp4;
    size_t cutlass_mxfp4_bytes = 0;
    bool use_mxfp4 = false;

    // --- Q4_K_M direct INT8 IMMA cache (Phase 2C infrastructure) ---
    // Populated at load-time by mmq_q4k_imma_reorder() when
    // gemm.q4k_imma_enabled = true. Consumed by mmq_q4k_imma_tile().
    // Three device buffers per entry:
    //   w_sym_s8 [N, K]      int8  symmetric-shifted (q - 8)
    //   eff_alpha [N, K/32]  FP16  d_super · sc[j]
    //   eff_beta  [N, K/32]  FP16  8·d_super·sc[j] − dmin_super·m[j]
    // Decode identity: α·q_sym + β  ≡  d·sc·q − dmin·m.
    //
    // The Phase 2C dispatcher (separate PR) gates entries on
    //   M ≥ 1024 && dense && Q4_K_M && !fp16_cache_hit
    // Off by default until E2E A/B against dense Q4_K_M models lands.
    struct Q4kImmaCacheEntry {
        int8_t* w_sym_s8 = nullptr;
        __half* eff_alpha = nullptr;
        __half* eff_beta = nullptr;
        int N = 0;
        int K = 0;
    };
    std::unordered_map<const void*, Q4kImmaCacheEntry> q4k_imma;
    size_t q4k_imma_bytes = 0;
    bool use_q4k_imma = false;

    // Dual-path mode: FP8 attention + NVFP4 FFN
    bool dual_path_quant = false;
};

// Per-call state for GraphExecutor::run_moe_ffn(). Bundles the locals that
// were previously captured by the monolithic body so each MoE phase helper
// can take a single MoeFfnContext& instead of a 20-arg parameter list.
// Populated by moe_ffn_phase1/2/…; subsequent phases read/mutate it.
struct MoeFfnContext {
    // Shape / dtype parameters
    int n = 0;
    int d = 0;
    int ne = 0;
    int top_k = 0;
    int eff = 0;
    int expanded = 0;
    float eps = 0.f;
    size_t es = 0;

    // Tensor views (point at hidden_, residual_, norm_out_ for the current step)
    Tensor h{};
    Tensor r{};
    Tensor no{};

    // Path-selection flags computed during setup/routing
    bool nvfp4_covers_layer = false;
    bool will_skip_residual_copy = false;
    bool gemma4_fp32_norm = false;
    bool moe_use_fp32_residual = false;
    bool moe_fused_norm_q8 = false;
    bool fp32_down_active = false;
    bool fp32_gate_logits_ready = false;
    bool will_decode_fast = false;
    bool non_gated_experts = false;
    bool use_packed_dequant = false;
    QType up_qtype = QType::F16;

    // Routing result + transient buffers carried across phases
    MoeRoutingResult routing{};
    void* fp32_down_buf = nullptr;
    bool residual_fused = false;  // true when decode-fast / fused scatter already added residual
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
    size_t workspace_estimate() const;

    // Run the full forward pass and return the sampled token ID.
    int32_t forward(const InferenceState& state, cudaStream_t stream = nullptr);

    // Batched forward: returns one sampled token per sequence.
    std::vector<int32_t> forward_batch(const InferenceState& state, cudaStream_t stream = nullptr);

    // Run the forward pass but return raw logits instead of sampling.
    // logits_out will be a view into the internal logits buffer.
    void forward_logits(const InferenceState& state, Tensor& logits_out, cudaStream_t stream = nullptr);

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
    [[nodiscard]] bool resize_workspace(int new_max_tokens, cudaStream_t stream);

    // Dual workspace for concurrent prefill/decode overlap.
    // allocate_decode_workspace: creates a second workspace for decode (up to max_batch tokens).
    // use_workspace(0) = prefill (default), use_workspace(1) = decode.
    bool allocate_decode_workspace(cudaStream_t stream, int max_batch = 1);
    void use_workspace(int slot);  // 0=prefill, 1=decode
    bool has_decode_workspace() const { return decode_workspace_ != nullptr; }
    int active_workspace() const { return active_workspace_; }
    int max_tokens() const { return max_tokens_; }

    // Capacity of the [n_heads, attn_seq, attn_seq] FP16 attn-scores workspace.
    // Engine's chunked-prefill path must clamp chunk_len so n × ctx_len ≤ cap².
    // Returns 0 if the buffer wasn't allocated (VRAM-constrained / WMMA fallback).
    int attn_scores_cap() const {
        return attn_scores_buf_ ? static_cast<int>(attn_scores_.shape[1]) : 0;
    }

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

    // Access the hidden state buffer after forward_logits().
    // Returns [max_tokens, d_model] FP16 on device. Use view_tokens() to get [n, d_model].
    const Tensor& hidden_state() const { return hidden_; }

    // Public view_tokens wrapper for external callers.
    Tensor view_hidden(int n_tokens) const { return view_tokens(hidden_, n_tokens); }

private:
    // Phases of pre_dequant_weights(), extracted for readability.
    // Cross-phase state: remaining_budget is reduced by each FP16/FP8/NVFP4
    // pass; cfg is const reference to model config.
    void pre_dequant_phase0_promote_nvfp4_sidecars_(const ModelConfig& cfg, cudaStream_t stream);
    void pre_dequant_phase0b_register_cutlass_nvfp4_(const ModelConfig& cfg, cudaStream_t stream);
    void pre_dequant_phase1_fp16_cache_(const ModelConfig& cfg, const VRAMBudget& budget,
                                        size_t& remaining_budget, cudaStream_t stream);
    void pre_dequant_phase2_fp8_cache_(const ModelConfig& cfg, const VRAMBudget& budget,
                                       size_t& remaining_budget, cudaStream_t stream);
    void pre_dequant_phase3_nvfp4_decode_(const ModelConfig& cfg, const VRAMBudget& budget,
                                          size_t& remaining_budget, cudaStream_t stream);
    void pre_dequant_phase3c_standalone_mxfp4_(const ModelConfig& cfg, cudaStream_t stream);
    void pre_dequant_phase4_tensor_registry_(const ModelConfig& cfg, cudaStream_t stream);

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
    int max_logit_tokens_ = 0;     // max tokens needing LM head projection (= max_batch_size)
    int cur_n_tokens_ = 0;         // set by forward_logits for use by run_ffn
    int cur_decode_step_ = 0;      // set by forward_logits for debug dump tagging
    bool cur_force_fp16_ = false;  // set by forward_logits, bypasses FP8 GEMM paths
    bool cur_per_row_lm_ = false;  // set by forward_logits, per-row Q8_1 LM head

    // Programmatic Dependent Launch: when true, custom kernels have the PDL
    // attribute set so the GPU can overlap tail of one kernel with head of next.
    bool use_pdl_ = false;

    // --- Persistent GPU workspace (always valid, not reconfigured) ---
    void* persistent_workspace_ = nullptr;
    size_t persistent_workspace_size_ = 0;

    // Persistent activation tensors (views into persistent_workspace_)
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

    // Weight caches (FP16, FP8, NVFP4, CUTLASS NVFP4/MXFP4, fused KV/gate+up)
    WeightCaches wcache_;

    // Mode flags mirrored from wcache_ for PlanHints (hints_ is the Phase 5 source of truth).
    PlanHints hints_;

    // WeightRegistry: parallel handle store (Phase 2+ shim, populated alongside wcache_)
    WeightRegistry registry_;

    // Quantization scratch buffers (FP8 act, CUTLASS act, dp4a, dequant, split-K)
    QuantScratch qscratch_;

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

    // --- Dual workspace for concurrent prefill/decode overlap ---
    // Slot 0 (default): main workspace (prefill, sized for max_tokens)
    // Slot 1: decode workspace (sized for up to decode_max_batch_ tokens)
    void* decode_workspace_ = nullptr;         // persistent buf for decode
    void* decode_shared_workspace_ = nullptr;  // shared buf for decode
    size_t decode_persistent_size_ = 0;
    size_t decode_shared_size_ = 0;
    int decode_max_batch_ = 1;  // max decode batch size this workspace supports
    int active_workspace_ = 0;

    // Saved prefill workspace pointers (restored when switching back)
    struct SavedWorkspace {
        void* persistent;
        size_t persistent_size;
        void* shared;
        size_t shared_size;
        int shared_max_tokens;
        Tensor hidden, residual, norm_out, logits;
        void* fp32_accum;
        Tensor fp32_hidden;
    };
    SavedWorkspace saved_prefill_ws_;

    // --- Layer offload manager (non-owning, set by engine) ---
    LayerOffloadManager* offload_mgr_ = nullptr;

    // --- Allocation and configuration methods ---

    [[nodiscard]] bool allocate_persistent_workspace(int max_tokens);
    [[nodiscard]] bool allocate_shared_workspace(int max_tokens);
    void allocate_auxiliary_buffers(
        bool skip_batch_dequant = false);  // dequant scratch, MoE staging, routing buffers
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
