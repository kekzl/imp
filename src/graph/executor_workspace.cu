#include "graph/executor.h"
#include "graph/executor_kernels.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#ifdef IMP_USE_CUTLASS
#include "compute/gemm_cutlass.h"
#include "compute/gemm_cutlass_sm120.h"
#include "compute/attention_cutlass_fmha.h"
#endif
#include "compute/activation.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "quant/fp8_quant.h"
#include "quant/nvfp4_gemm.h"
#include "compute/gemm_cublaslt_nvfp4.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "memory/vram_allocator.h"
#include "runtime/pdl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifdef __CUDA_FP8_TYPES_EXIST__
#include <cuda_fp8.h>
#endif
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

namespace imp {

// Helper: round up to 256-byte alignment (used throughout for workspace layout).
static inline size_t align256(size_t x) { return (x + 255) & ~size_t(255); }

// Helper: saturating budget deduction (avoids underflow on size_t).
static inline void deduct_budget(size_t& budget, size_t amount) {
    budget = (budget > amount) ? (budget - amount) : 0;
}

// Helper: create a 2D Tensor view at `ptr` and advance ptr by `aligned_sz`.
// Used by configure_*_workspace() and allocate_persistent_workspace() to lay out
// tensors in a contiguous buffer.
static Tensor make_workspace_tensor(char*& ptr, DType dtype, int64_t rows, int64_t cols, size_t aligned_sz) {
    int64_t shape[2] = {rows, cols};
    Tensor t(ptr, dtype, 2, shape, true);
    ptr += aligned_sz;
    return t;
}

// Helper: allocate GPU memory through VRAMAllocator if available, else cudaMalloc.
// Frees through the same path. Used for large persistent allocations.
static void* vram_alloc(VRAMAllocator* alloc, size_t bytes, const char* tag) {
    if (bytes == 0) return nullptr;
    if (alloc) return alloc->allocate(bytes, tag);
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) return nullptr;
    return ptr;
}

static void vram_free(VRAMAllocator* alloc, void* ptr) {
    if (!ptr) return;
    if (alloc) alloc->free(ptr);
    else cudaFree(ptr);
}

// Helper: create a fused weight pair by concatenating two FP16 cached weights.
// Used for fused KV and fused gate+up weight creation in Phase 1.
// Returns true if the fused weight was created, false if skipped/failed.
// Sets should_stop=true when budget is exhausted or allocation fails (caller should break).
static bool create_fused_weight_pair(
    const Tensor& w_a, const Tensor& w_b,
    const std::unordered_map<const void*, Tensor>& fp16_cache,
    VRAMAllocator* allocator,
    size_t& total_cache_bytes, size_t remaining_budget,
    cudaStream_t stream,
    std::unordered_map<int, Tensor>& out_map, int layer_idx,
    bool& should_stop)
{
    should_stop = false;
    if (!w_a.data || !w_b.data) return false;
    // Both must be in FP16 cache
    auto it_a = fp16_cache.find(w_a.data);
    auto it_b = fp16_cache.find(w_b.data);
    if (it_a == fp16_cache.end() || it_b == fp16_cache.end()) return false;

    int a_rows = static_cast<int>(w_a.shape[0]);
    int K = static_cast<int>(w_a.shape[1]);
    size_t one_sz = static_cast<size_t>(a_rows) * K * sizeof(half);

    // Respect VRAM budget — on WSL2/WDDM, cudaMalloc silently spills to
    // shared (system) memory beyond physical VRAM, causing massive slowdowns.
    if (total_cache_bytes + 2 * one_sz > remaining_budget) {
        should_stop = true;
        return false;
    }

    void* fused_buf = vram_alloc(allocator, 2 * one_sz, "fp16_weight_cache");
    if (!fused_buf) {
        should_stop = true;
        return false;
    }

    cudaMemcpyAsync(fused_buf, it_a->second.data, one_sz,
                     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(static_cast<char*>(fused_buf) + one_sz,
                     it_b->second.data, one_sz,
                     cudaMemcpyDeviceToDevice, stream);

    int64_t shape[2] = {2 * a_rows, static_cast<int64_t>(K)};
    out_map[layer_idx] = Tensor(fused_buf, DType::FP16, 2, shape, true);
    total_cache_bytes += 2 * one_sz;
    return true;
}

// Helper: iterate all dense layer weights in priority order (attention first,
// then SSM/shared/FFN). Used by Phases 2-3 of pre_dequant_weights to collect
// weights for FP8/NVFP4 caching with a consistent priority ordering.
// The callback receives (weight_tensor, qtype) for each eligible weight.
template <typename Fn>
static void for_each_dense_weight(const Model& model, const ModelConfig& cfg, Fn&& fn) {
    // Pass 1: attention weights (critical for cuBLAS prefill)
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model.layer(i);
        fn(L.wq, L.wq_qtype);
        fn(L.wk, L.wk_qtype);
        fn(L.wv, L.wv_qtype);
        fn(L.wo, L.wo_qtype);
    }
    // Pass 2: SSM, shared experts, and dense FFN
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model.layer(i);
        fn(L.ssm_in, L.ssm_in_qtype);
        fn(L.ssm_out, L.ssm_out_qtype);
        fn(L.w_gate_shared, L.w_gate_shared_qtype);
        fn(L.w_up_shared, L.w_up_shared_qtype);
        fn(L.w_down_shared, L.w_down_shared_qtype);
        fn(L.w_gate, L.w_gate_qtype);
        fn(L.w_up, L.w_up_qtype);
        fn(L.w_down, L.w_down_qtype);
    }
}

// ---------------------------------------------------------------------------
// GraphExecutor lifetime
// ---------------------------------------------------------------------------

GraphExecutor::~GraphExecutor() {
    free_buffers();
}

bool GraphExecutor::init(const Model& model, DType compute_dtype, bool use_pdl,
                         int max_batch_size, int max_seq_len, bool use_fp8_prefill,
                         int use_nvfp4_decode, bool use_mxfp4_prefill) {
    if (initialized_) {
        free_buffers();
    }

    model_ = &model;
    compute_dtype_ = compute_dtype;
    norm_w_off_ = model.config().norm_weight_offset;
    use_pdl_ = use_pdl;
    wcache_.use_fp8 = use_fp8_prefill;
    wcache_.nvfp4_decode_mode = use_nvfp4_decode;
    wcache_.use_mxfp4 = use_mxfp4_prefill;

    const auto& cfg = model.config();

    // Detect model features for workspace sizing
    has_moe_ = (cfg.n_experts > 0 && cfg.n_experts_active > 0);
    has_ssm_ = (cfg.ssm_inner_size > 0);
    has_gdn_ = false;  // detected from tensor presence below
    has_dense_ffn_ = (cfg.d_ff > 0);

    // Compute max expert FFN hidden dim from actual packed tensor shapes.
    // cfg.expert_d_ff may not match the actual tensor dimensions (e.g. Nemotron-H).
    max_expert_eff_ = cfg.expert_d_ff;
    if (has_moe_) {
        for (int li = 0; li < cfg.n_layers; li++) {
            const auto& L = model.layer(li);
            // gate/up packed: shape [n_experts, expert_d_ff, d_model]
            for (const auto* p : {&L.expert_gate_packed, &L.expert_up_packed}) {
                if (p->data && p->ndim >= 3)
                    max_expert_eff_ = std::max(max_expert_eff_, static_cast<int>(p->shape[1]));
            }
            // down packed: shape [n_experts, d_model, expert_d_ff]
            if (L.expert_down_packed.data && L.expert_down_packed.ndim >= 3)
                max_expert_eff_ = std::max(max_expert_eff_, static_cast<int>(L.expert_down_packed.shape[2]));
        }
        if (max_expert_eff_ != cfg.expert_d_ff) {
            IMP_LOG_WARN("expert_d_ff mismatch: config=%d, actual packed tensors=%d — using %d",
                         cfg.expert_d_ff, max_expert_eff_, max_expert_eff_);
        }
    }

    // Use engine-provided max_seq_len if given, otherwise fall back to model config.
    int effective_seq_len = (max_seq_len > 0) ? max_seq_len : cfg.max_seq_len;
    max_tokens_ = std::min(effective_seq_len, 4096);
    if (max_tokens_ <= 0) {
        max_tokens_ = 4096;
    }

    // Cap max_tokens for hybrid MoE+SSM/GDN models to limit workspace.
    // SSM state + cuBLAS S-matrix + workspace can exhaust 32 GB VRAM.
    if (has_ssm_ && (has_moe_ || has_gdn_)) {
        int capped = has_moe_ ? 256 : 512;  // MoE tighter, dense GDN can afford more
        if (max_tokens_ > capped) {
            IMP_LOG_INFO("executor_workspace.cu:%d: Capping max_tokens %d → %d for SSM/GDN hybrid",
                         __LINE__, max_tokens_, capped);
            max_tokens_ = capped;
        }
    }

    // Logits buffer only needs to hold tokens that require LM head projection:
    // - Prefill: 1 (last token only)
    // - Decode:  n_sequences (one per batch slot)
    max_logit_tokens_ = std::max(max_batch_size, 1);

    // Compute shared workspace sizes (no allocation — deferred to allocate_workspaces()).
    // Deferring GPU allocation maximizes VRAM available for expert weight upload.
    compute_shared_sizes(max_tokens_);

    // Build SSM layer index mapping
    if (has_ssm_) {
        ssm_layer_map_.resize(cfg.n_layers, -1);
        int ssm_idx = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            if (model_->layer(i).ssm_in.data != nullptr) {
                ssm_layer_map_[i] = ssm_idx++;
            }
        }
        IMP_LOG_INFO("SSM layers: %d out of %d total", ssm_idx, cfg.n_layers);
    }

    // Build GDN layer index mapping (Gated DeltaNet, e.g., Qwen3.5)
    {
        gdn_layer_map_.resize(cfg.n_layers, -1);
        int gdn_idx = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            if (model_->layer(i).gdn_gate.data != nullptr) {
                gdn_layer_map_[i] = gdn_idx++;
            }
        }
        if (gdn_idx > 0) {
            has_gdn_ = true;
            IMP_LOG_INFO("GDN layers: %d out of %d total", gdn_idx, cfg.n_layers);
        }
    }

    // Enable Programmatic Dependent Launch on custom kernels if requested.
    if (use_pdl_ && pdl::is_available()) {
        pdl::enable(reinterpret_cast<const void*>(&elementwise_add_fp16_kernel));
        pdl::enable(reinterpret_cast<const void*>(&elementwise_add_fp32_kernel));
        pdl::enable(reinterpret_cast<const void*>(&write_kv_cache_kernel));
        pdl::enable(reinterpret_cast<const void*>(&write_kv_cache_fused_kernel));
        pdl::enable(reinterpret_cast<const void*>(&write_kv_cache_rope_fused_kernel));
        pdl::enable(reinterpret_cast<const void*>(&fp16_to_fp32_kernel));
        pdl::enable(reinterpret_cast<const void*>(&fp32_to_fp16_kernel));
        // Register compute kernels for PDL overlap (run between GEMMs in hot path)
        layernorm_pdl_register();
        rope_pdl_register();
        activation_pdl_register();
        gemv_pdl_register();
        nvfp4_gemv_pdl_register();
        IMP_LOG_INFO("PDL enabled on executor + compute + GEMV kernels");
    } else if (use_pdl_) {
        IMP_LOG_WARN("PDL requested but not available on this device/CUDA version");
        use_pdl_ = false;
    }

    // Precompute YaRN correction dimensions if enabled
    if (cfg.yarn_ext_factor > 0.0f) {
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
        int n_dims = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
        int n_ctx_orig = cfg.rope_n_ctx_orig > 0 ? cfg.rope_n_ctx_orig : cfg.max_seq_len;
        rope_yarn_corr_dims(n_dims, n_ctx_orig, cfg.rope_theta,
                            cfg.yarn_beta_fast, cfg.yarn_beta_slow, yarn_corr_dims_);
        IMP_LOG_INFO("YaRN corr_dims: [%.1f, %.1f] (n_dims=%d, n_ctx_orig=%d)",
                     yarn_corr_dims_[0], yarn_corr_dims_[1], n_dims, n_ctx_orig);
    }

    // Pre-compute LongRoPE inverse frequencies if enabled (Phi-4)
    if (!cfg.rope_short_factor.empty() && !cfg.rope_long_factor.empty()) {
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
        int rd = (cfg.rope_dim > 0) ? cfg.rope_dim : hd;
        int pairs = rd / 2;
        longrope_n_pairs_ = pairs;
        longrope_orig_max_pos_ = cfg.rope_scaling_orig_max_pos;

        // inv_freq[i] = 1.0 / (factor[i] * theta^(2i/rd))
        std::vector<float> short_freqs(pairs), long_freqs(pairs);
        for (int i = 0; i < pairs; i++) {
            float base_freq = 1.0f / std::pow(cfg.rope_theta, (2.0f * i) / static_cast<float>(rd));
            short_freqs[i] = base_freq / cfg.rope_short_factor[i];
            long_freqs[i]  = base_freq / cfg.rope_long_factor[i];
        }

        cudaError_t e1 = cudaMalloc(&longrope_short_freqs_, pairs * sizeof(float));
        cudaError_t e2 = cudaMalloc(&longrope_long_freqs_,  pairs * sizeof(float));
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate LongRoPE frequency buffers: %s",
                          cudaGetErrorString(e1 != cudaSuccess ? e1 : e2));
            if (longrope_short_freqs_) { cudaFree(longrope_short_freqs_); longrope_short_freqs_ = nullptr; }
            if (longrope_long_freqs_)  { cudaFree(longrope_long_freqs_);  longrope_long_freqs_ = nullptr; }
            return false;
        }
        cudaMemcpy(longrope_short_freqs_, short_freqs.data(), pairs * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(longrope_long_freqs_,  long_freqs.data(),  pairs * sizeof(float), cudaMemcpyHostToDevice);

        IMP_LOG_INFO("LongRoPE: %d freq pairs, orig_max_pos=%d", pairs, longrope_orig_max_pos_);
    }

    initialized_ = true;

    IMP_LOG_INFO("GraphExecutor initialized: max_tokens=%d, d_model=%d, "
                 "n_layers=%d, dtype=%s, pdl=%s",
                 max_tokens_, cfg.d_model, cfg.n_layers,
                 dtype_name(compute_dtype_),
                 use_pdl_ ? "on" : "off");
    return true;
}

// ---------------------------------------------------------------------------
// Phase 2: allocate all GPU workspace buffers (called after weight upload)
// ---------------------------------------------------------------------------

bool GraphExecutor::allocate_workspaces(bool experts_on_host) {
    if (!initialized_ || !model_) return false;

    if (!allocate_persistent_workspace(max_tokens_)) {
        IMP_LOG_ERROR("Persistent workspace allocation failed — cannot run inference");
        return false;
    }
    if (!allocate_shared_workspace(max_tokens_)) {
        IMP_LOG_ERROR("Shared workspace allocation failed — cannot run inference");
        return false;
    }
    allocate_auxiliary_buffers(/*skip_batch_dequant=*/experts_on_host);

    return true;
}

size_t GraphExecutor::workspace_estimate() const {
    if (!model_) return 0;
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    size_t es = dtype_size(compute_dtype_);


    // Persistent: hidden + residual + norm_out + logits
    size_t persistent = 3 * align256(static_cast<size_t>(max_tokens_) * d * es)
                      + align256(static_cast<size_t>(max_logit_tokens_) * cfg.vocab_size * sizeof(float));

    // Shared: max of phases (already computed in compute_shared_sizes)
    size_t shared = std::max({attn_shared_size_, ffn_shared_size_,
                              moe_shared_size_, ssm_shared_size_});

    // S-matrix is NOT included here — it's optional (flash attention fallback works).
    // This maximizes VRAM available for expert layers during weight upload.
    // S-matrix is allocated opportunistically from remaining VRAM.

    // FP32 accumulator for post-norm models (Gemma-3): 1 × max_tokens × d_model × 4
    bool has_post_norms = (cfg.norm_placement == NormPlacement::POST_NORM);
    size_t fp32_accum = has_post_norms ? align256(static_cast<size_t>(max_tokens_) * d * sizeof(float)) : 0;

    // Auxiliary: compute real estimate from individual buffer sizes
    size_t auxiliary = 0;

    // Dequant scratch: max weight matrix elements × sizeof(half)
    {
        size_t max_elems = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data) max_elems = std::max(max_elems, static_cast<size_t>(w->numel()));
            }
        }
        auxiliary += max_elems * sizeof(uint16_t);
    }

    // Sampling result (ARGMAX_SCRATCH_BYTES ~16 KiB) + MMVQ scratch + split-K scratch
    int nh_est = cfg.n_heads;
    int hd_est = cfg.head_dim > 0 ? cfg.head_dim : (d / nh_est);
    auxiliary += 16 * 1024;  // sampling
    auxiliary += 256 * 1024;  // MMVQ scratch (conservative)
    auxiliary += static_cast<size_t>(max_logit_tokens_) * nh_est * 32 * (2 + hd_est) * sizeof(float);  // split-K

    // S-matrix for cuBLAS attention fallback — only needed when CUTLASS FMHA
    // is unavailable or unsupported (e.g., softcap, sliding window).
    // Skip for MoE-heavy models where VRAM is tight: WMMA/CUTLASS FMHA
    // doesn't need the S-matrix. This saves up to 256 MiB.
    bool is_moe = (cfg.n_experts > 0 && cfg.n_experts_active > 0);
    if (!is_moe) {
        auxiliary += std::min(static_cast<size_t>(nh_est) * max_tokens_ * max_tokens_ * sizeof(half),
                              static_cast<size_t>(256) << 20);
    }

    // Safety margin for FP8 act buffers, misc (reduced for MoE to save VRAM)
    auxiliary += is_moe ? (8ULL << 20) : (32ULL << 20);

#ifdef IMP_USE_CUTLASS
    // CUTLASS FMHA workspace (LSE buffer + kernel cooperative workspace)
    // Skip for MoE models where prefill uses cuBLAS (compute-light attention)
    if (!is_moe) {
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads);
        auxiliary += cutlass_fmha_workspace_estimate(1, max_tokens_, cfg.n_heads, hd);
    }
#endif

    return persistent + shared + fp32_accum + auxiliary;
}

// ---------------------------------------------------------------------------
// Unified workspace allocation
// ---------------------------------------------------------------------------

void GraphExecutor::compute_shared_sizes(int max_tokens) {
    const auto& cfg = model_->config();
    int d   = cfg.d_model;
    int ff  = cfg.d_ff;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);



    // Attention phase: q, k+v (contiguous for batched GEMM), attn_out, proj_out
    // Check for Q+Gate interleaving (Qwen3.5): Q projection output is 2x larger
    // than standard Q when an attention output gate is present.
    int max_q_out = nh * hd;
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& ly = model_->layer(i);
        if (ly.wq.data) {
            int q_dim = static_cast<int>(ly.wq.shape[0]);
            if (q_dim > max_q_out) max_q_out = q_dim;
        }
    }
    size_t kv_raw = static_cast<size_t>(max_tokens) * nkv * hd * es;
    attn_shared_size_ = align256(static_cast<size_t>(max_tokens) * nh * hd * es)    // q (de-interleaved)
                       + align256(2 * kv_raw)                                       // k+v contiguous
                       + align256(static_cast<size_t>(max_tokens) * nh * hd * es)   // attn_out
                       + align256(static_cast<size_t>(max_tokens) * d * es)         // proj_out
                       + (max_q_out > nh * hd
                          ? align256(static_cast<size_t>(max_tokens) * max_q_out * es)  // qv_full (Q+Gate)
                          : 0);

    // Dense FFN phase: gate, up, swiglu, ffn_out
    if (has_dense_ffn_ && ff > 0) {
        ffn_shared_size_ = align256(static_cast<size_t>(max_tokens) * ff * es)   // gate_out
                          + align256(static_cast<size_t>(max_tokens) * ff * es)  // up_out
                          + align256(static_cast<size_t>(max_tokens) * ff * es)  // swiglu_out
                          + align256(static_cast<size_t>(max_tokens) * d * es);  // ffn_out
    }

    // MoE phase
    if (has_moe_) {
        int ne    = cfg.n_experts;
        int top_k = cfg.n_experts_active;
        int eff   = max_expert_eff_;
        int expanded = max_tokens * top_k;

        moe_shared_size_ = align256(static_cast<size_t>(max_tokens) * ne * sizeof(float))  // gate_logits
                          + align256(static_cast<size_t>(expanded) * d * es)                // gathered
                          + align256(static_cast<size_t>(expanded) * eff * es)              // expert_gate
                          + align256(static_cast<size_t>(expanded) * eff * es)              // expert_up
                          + align256(static_cast<size_t>(expanded) * eff * es)              // expert_swiglu
                          + align256(static_cast<size_t>(expanded) * d * es)                // expert_down
                          + align256(static_cast<size_t>(max_tokens) * d * sizeof(float));  // scatter_out
    }

    // SSM phase
    if (has_ssm_) {
        int inner = cfg.ssm_inner_size;
        int n_groups = cfg.ssm_group_count;
        int state_size = cfg.ssm_state_size;
        int n_heads = cfg.ssm_dt_rank;
        int conv_channels = inner + 2 * n_groups * state_size;
        int ssm_in_dim = inner + conv_channels + n_heads;

        size_t proj_elem_size = has_gdn_ ? sizeof(float) : es;
        ssm_shared_size_ = align256(static_cast<size_t>(max_tokens) * ssm_in_dim * proj_elem_size) // proj (FP32 for GDN)
                          + align256(static_cast<size_t>(max_tokens) * conv_channels * es)   // xBC
                          + align256(static_cast<size_t>(max_tokens) * inner * es)           // y
                          + align256(static_cast<size_t>(max_tokens) * inner * es)           // z
                          + align256(static_cast<size_t>(max_tokens) * d * es)               // out
                          + align256(static_cast<size_t>(max_tokens) * n_heads * (has_gdn_ ? 2 : 1) * es); // dt (2x for GDN: alpha + beta)
    }
}

bool GraphExecutor::allocate_persistent_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int v = cfg.vocab_size;
    size_t es = dtype_size(compute_dtype_);



    size_t hidden_sz   = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t residual_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t norm_out_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t logits_sz   = align256(static_cast<size_t>(max_logit_tokens_) * v * sizeof(float));

    size_t total = hidden_sz + residual_sz + norm_out_sz + logits_sz;

    persistent_workspace_ = vram_alloc(vram_alloc_, total, "persistent_workspace");
    if (!persistent_workspace_) {
        IMP_LOG_ERROR("Failed to allocate persistent workspace (%.1f MiB)",
                      total / (1024.0 * 1024.0));
        return false;
    }
    persistent_workspace_size_ = total;

    char* ptr = static_cast<char*>(persistent_workspace_);

    hidden_   = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d, hidden_sz);
    residual_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d, residual_sz);
    norm_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d, norm_out_sz);

    {
        int64_t shape[2] = {static_cast<int64_t>(max_logit_tokens_), static_cast<int64_t>(v)};
        logits_ = Tensor(ptr, DType::FP32, 2, shape, true);
        ptr += logits_sz;
    }

    IMP_LOG_INFO("Persistent workspace: %.2f MiB (hidden+residual+norm+logits)",
                 total / (1024.0 * 1024.0));

    // FP32 residual accumulator for post-norm architectures (Gemma-3).
    if (cfg.norm_placement == NormPlacement::POST_NORM) {
        size_t fp32_sz = align256(static_cast<size_t>(max_tokens) * d * sizeof(float));
        cudaError_t e2 = cudaMalloc(&fp32_accum_buf_, fp32_sz);
        if (e2 == cudaSuccess) {
            int64_t shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(d)};
            fp32_hidden_ = Tensor(fp32_accum_buf_, DType::FP32, 2, shape, true);
            IMP_LOG_INFO("FP32 residual accumulator: %.2f MiB (post-norm architecture)",
                         fp32_sz / (1024.0 * 1024.0));
        } else {
            IMP_LOG_WARN("Failed to allocate FP32 accumulator (%zu bytes): %s — falling back to FP16",
                         fp32_sz, cudaGetErrorString(e2));
        }
    }
    return true;
}

bool GraphExecutor::allocate_shared_workspace(int max_tokens) {
    size_t max_shared = std::max({attn_shared_size_, ffn_shared_size_,
                                  moe_shared_size_, ssm_shared_size_});
    if (max_shared == 0) return true;  // no workspace needed

    shared_workspace_ = vram_alloc(vram_alloc_, max_shared, "shared_workspace");
    if (!shared_workspace_) {
        // Shared workspace is critical for GEMV scratch buffers. Fall back to
        // raw cudaMalloc bypassing headroom (Nemotron-30B leaves <headroom free).
        IMP_LOG_WARN("Shared workspace: allocator rejected (%.1f MiB), trying raw cudaMalloc",
                      max_shared / (1024.0 * 1024.0));
        cudaError_t err = cudaMalloc(&shared_workspace_, max_shared);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate shared workspace (%.1f MiB)",
                          max_shared / (1024.0 * 1024.0));
            return false;
        }
    }
    shared_workspace_size_ = max_shared;
    shared_workspace_max_tokens_ = max_tokens;

    IMP_LOG_INFO("Shared workspace: %.2f MiB = max(attn=%.1f, ffn=%.1f, moe=%.1f, ssm=%.1f MiB) "
                 "— saved %.2f MiB vs separate allocation",
                 max_shared / (1024.0 * 1024.0),
                 attn_shared_size_ / (1024.0 * 1024.0),
                 ffn_shared_size_ / (1024.0 * 1024.0),
                 moe_shared_size_ / (1024.0 * 1024.0),
                 ssm_shared_size_ / (1024.0 * 1024.0),
                 (attn_shared_size_ + ffn_shared_size_ + moe_shared_size_ + ssm_shared_size_
                  - max_shared) / (1024.0 * 1024.0));

    // Pre-allocate MoE routing buffers (separate from shared workspace)
    if (has_moe_) {
        const auto& cfg = model_->config();
        moe_.routing_buffers.allocate(max_tokens, cfg.n_experts, cfg.n_experts_active);
    }
    return true;
}

void GraphExecutor::allocate_auxiliary_buffers(bool skip_batch_dequant) {
    const auto& cfg = model_->config();

    // Dequant scratch buffer for on-the-fly weight dequantization.
    {
        size_t max_weight_elems = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data) max_weight_elems = std::max(max_weight_elems,
                                                          static_cast<size_t>(w->numel()));
            }
        }
        if (max_weight_elems > 0) {
            qscratch_.dequant_size = max_weight_elems * sizeof(uint16_t);
            qscratch_.dequant = vram_alloc(vram_alloc_, qscratch_.dequant_size, "dequant_scratch");
            if (!qscratch_.dequant) {
                IMP_LOG_ERROR("Failed to allocate dequant scratch (%.1f MiB)",
                              qscratch_.dequant_size / (1024.0 * 1024.0));
                qscratch_.dequant_size = 0;
            } else {
                IMP_LOG_INFO("Dequant scratch buffer: %.2f MiB",
                             qscratch_.dequant_size / (1024.0 * 1024.0));
            }
        }
    }

    // Sampling result buffer: sized to hold the argmax result plus the
    // multi-block partial reduction scratch (ARGMAX_SCRATCH_BYTES).
    {
        cudaError_t err = cudaMalloc(&d_sample_result_, ARGMAX_SCRATCH_BYTES);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to allocate sampling result buffer: %s",
                          cudaGetErrorString(err));
            d_sample_result_ = nullptr;
        }
    }

    // Pinned host buffer for async sampling D2H copy (avoids stack-variable sync)
    if (!h_sample_pinned_ && d_sample_result_) {
        cudaError_t err = cudaHostAlloc(&h_sample_pinned_, sizeof(int32_t), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("cudaHostAlloc for sample pinned buffer failed: %s",
                         cudaGetErrorString(err));
            h_sample_pinned_ = nullptr;
        }
    }

    // MMVQ (dp4a) scratch buffers for quantized input vectors.
    // Find the max Q8_1 block count needed across all uses:
    //   1. Dense GEMV: max_k / 32 blocks (one input vector)
    //   2. MoE down projection: top_k * expert_d_ff / 32 blocks (per-expert quantized activations)
    {
        int max_k = 0;
        int max_moe_down_blocks = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data && w->ndim >= 2) {
                    max_k = std::max(max_k, static_cast<int>(w->shape[1]));
                }
            }
            // MoE expert weight inner dims
            if (L.expert_up_packed.data && L.expert_up_packed.ndim >= 3) {
                max_k = std::max(max_k, static_cast<int>(L.expert_up_packed.shape[2]));
            }
            if (L.expert_down_packed.data && L.expert_down_packed.ndim >= 3) {
                int down_k = static_cast<int>(L.expert_down_packed.shape[2]);
                max_k = std::max(max_k, down_k);
                // MoE down projection quantizes top_k expert activations contiguously
                max_moe_down_blocks = std::max(max_moe_down_blocks,
                    cfg.n_experts_active * (down_k / 32));
            }
            if (L.expert_gate_packed.data && L.expert_gate_packed.ndim >= 3) {
                max_k = std::max(max_k, static_cast<int>(L.expert_gate_packed.shape[2]));
            }
        }
        int max_blocks = std::max(max_k / 32, max_moe_down_blocks);
        if (max_blocks > 0) {
            qscratch_.q8_1_max_blocks = max_blocks;
            size_t q8_1_sz = static_cast<size_t>(qscratch_.q8_1_max_blocks) * sizeof(block_q8_1);
            size_t d8_sz = static_cast<size_t>(qscratch_.q8_1_max_blocks) * sizeof(float);
            cudaError_t err1 = cudaMalloc(&qscratch_.q8_1_buf, q8_1_sz);
            cudaError_t err2 = cudaMalloc(reinterpret_cast<void**>(&qscratch_.d8_buf), d8_sz);
            if (err1 != cudaSuccess || err2 != cudaSuccess) {
                IMP_LOG_WARN("Failed to allocate MMVQ scratch buffers, dp4a path disabled");
                if (qscratch_.q8_1_buf) { cudaFree(qscratch_.q8_1_buf); qscratch_.q8_1_buf = nullptr; }
                if (qscratch_.d8_buf) { cudaFree(qscratch_.d8_buf); qscratch_.d8_buf = nullptr; }
                qscratch_.q8_1_max_blocks = 0;
            } else {
                IMP_LOG_INFO("MMVQ scratch buffers: %.2f KiB (q8_1) + %.2f KiB (d8), max_blocks=%d (max_k=%d, moe_down=%d)",
                             q8_1_sz / 1024.0, d8_sz / 1024.0, max_blocks, max_k, max_moe_down_blocks);
            }
        }
    }

    // Split-K paged attention scratch buffer.
    // Sized for max_batch_size * n_heads * max_splits * (2 + head_dim) floats.
    {
        int nh = cfg.n_heads;
        int hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / nh);
        // Size splits proportional to max context blocks, capped at 32
        int max_context_blocks = (max_tokens_ + kKVBlockSize - 1) / kKVBlockSize;
        int max_splits = std::min(32, std::max(1, max_context_blocks));
        int partial_stride = 2 + hd;
        int max_batch = max_logit_tokens_;  // = max_batch_size
        size_t sz = static_cast<size_t>(max_batch) * nh * max_splits * partial_stride * sizeof(float);
        cudaError_t err = cudaMalloc(&qscratch_.splitk, sz);
        if (err != cudaSuccess) {
            IMP_LOG_WARN("Failed to allocate split-K scratch (%zu bytes), split-K disabled", sz);
            qscratch_.splitk = nullptr;
            qscratch_.splitk_size = 0;
        } else {
            qscratch_.splitk_size = sz;
            IMP_LOG_INFO("Split-K paged attention scratch: %.2f KiB", sz / 1024.0);
        }
    }

    // cuBLAS attention S-matrix workspace: [n_heads, attn_seq, attn_seq] FP16
    // Used for prefill at medium sequence lengths (faster than WMMA flash attention
    // due to higher TC utilization in cuBLAS GEMM). Falls back to flash attention
    // for long sequences or when VRAM-constrained.
    if (!skip_batch_dequant) {
        int nh = cfg.n_heads;
        constexpr size_t kMaxAttnScoresMiB = 256;  // cap at 256 MiB
        size_t max_s_sz = kMaxAttnScoresMiB << 20;
        // max seq = sqrt(budget / (n_heads * sizeof(half)))
        int attn_seq = max_tokens_;
        size_t s_sz = static_cast<size_t>(nh) * attn_seq * attn_seq * sizeof(half);
        if (s_sz > max_s_sz) {
            attn_seq = static_cast<int>(std::sqrt(
                static_cast<double>(max_s_sz) / (nh * sizeof(half))));
            attn_seq = (attn_seq / 16) * 16;  // round down to multiple of 16
            if (attn_seq < 32) attn_seq = 0;  // too small to be useful
            s_sz = static_cast<size_t>(nh) * attn_seq * attn_seq * sizeof(half);
        }
        if (attn_seq > 0) {
            attn_scores_buf_ = vram_alloc(vram_alloc_, s_sz, "attn_scores");
            if (!attn_scores_buf_) {
                cudaError_t e = cudaGetLastError();
                IMP_LOG_WARN("Failed to allocate cuBLAS attention S-matrix (%.1f MiB): %s — "
                             "will fall back to WMMA attention for prefill",
                             s_sz / (1024.0 * 1024.0), cudaGetErrorString(e));
                attn_scores_buf_size_ = 0;
            } else {
                attn_scores_buf_size_ = s_sz;
                int64_t s_shape[3] = {static_cast<int64_t>(nh),
                                      static_cast<int64_t>(attn_seq),
                                      static_cast<int64_t>(attn_seq)};
                attn_scores_ = Tensor(attn_scores_buf_, DType::FP16, 3, s_shape, true);
                IMP_LOG_INFO("cuBLAS attention S-matrix: %.2f MiB (%d heads x %d x %d)",
                             s_sz / (1024.0 * 1024.0), nh, attn_seq, attn_seq);
            }
        }
    } else {
        IMP_LOG_INFO("cuBLAS attention S-matrix: skipped (VRAM-constrained, using WMMA/TCGEN05 fallback)");
    }

#ifdef IMP_USE_CUTLASS
    // CUTLASS FMHA workspace: pre-allocate LSE + kernel workspace at max dimensions.
    // This ensures the allocations are tracked in the VRAM budget instead of happening
    // lazily (which would cause untracked VRAM growth and potential shared memory swapping).
    {
        int fmha_nh = cfg.n_heads;
        int fmha_hd = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / fmha_nh);
        size_t fmha_bytes = cutlass_fmha_init_workspace(1, max_tokens_, fmha_nh, fmha_hd);
        if (fmha_bytes > 0) {
            IMP_LOG_INFO("CUTLASS FMHA workspace: %.2f MiB (LSE + kernel)",
                         fmha_bytes / (1024.0 * 1024.0));
        }
    }
#endif

    // MoE dequant and staging buffers
    if (has_moe_) {
        int d   = cfg.d_model;
        int eff = max_expert_eff_;

        // Dequant buffer: 1 expert slot
        {
            size_t expert_fp16_elems = static_cast<size_t>(eff) * d;
            size_t dequant_sz = expert_fp16_elems * sizeof(uint16_t);
            moe_.dequant_buf = vram_alloc(vram_alloc_, dequant_sz, "moe_dequant");
            if (!moe_.dequant_buf) {
                IMP_LOG_ERROR("Failed to allocate MoE dequant buffer (%zu bytes)", dequant_sz);
                moe_.dequant_buf_size = 0;
            } else {
                moe_.dequant_buf_size = dequant_sz;
                IMP_LOG_INFO("MoE dequant buffer: %.2f MiB (1 expert slot)",
                             dequant_sz / (1024.0 * 1024.0));
            }
        }

        // Staging buffer for host→device expert weight transfer
        size_t max_expert_raw = 0;
        {
            for (int li = 0; li < model_->n_layers(); li++) {
                const auto& L = model_->layer(li);
                auto check = [&](const Tensor& p, GGMLQuantType qt) {
                    if (!p.data || p.ndim < 3) return;
                    size_t rb = ggml_quant_row_bytes(qt, p.shape[2]);
                    size_t expert_raw = static_cast<size_t>(p.shape[1]) * rb;
                    max_expert_raw = std::max(max_expert_raw, expert_raw);
                };
                check(L.expert_up_packed, L.expert_up_qtype);
                check(L.expert_down_packed, L.expert_down_qtype);
                check(L.expert_gate_packed, L.expert_gate_qtype);
            }
            if (max_expert_raw > 0) {
                moe_.raw_staging_buf = vram_alloc(vram_alloc_, max_expert_raw, "moe_staging");
                if (!moe_.raw_staging_buf) {
                    IMP_LOG_ERROR("Failed to allocate MoE staging buffer (%zu bytes)", max_expert_raw);
                    moe_.raw_staging_size = 0;
                } else {
                    moe_.raw_staging_size = max_expert_raw;
                    IMP_LOG_INFO("MoE staging buffer: %.2f MiB (1 expert raw)",
                                 max_expert_raw / (1024.0 * 1024.0));
                }
            }
        }

        // LRU expert cache: keeps recently-used host experts on GPU.
        // Only allocated when some experts reside on host (not all fit in VRAM).
        if (max_expert_raw > 0) {
            bool has_host_experts = false;
            for (int li = 0; li < model_->n_layers(); li++) {
                const auto& L = model_->layer(li);
                if ((L.expert_up_packed.data && !L.expert_up_packed.on_device) ||
                    (L.expert_down_packed.data && !L.expert_down_packed.on_device) ||
                    (L.expert_gate_packed.data && !L.expert_gate_packed.on_device)) {
                    has_host_experts = true;
                    break;
                }
            }
            if (has_host_experts) {
                // Budget: proportional to free VRAM (15%) instead of flat cap.
                // KV cache + weight caches (FP8/NVFP4) need the remaining VRAM,
                // so expert cache must not over-commit.
                size_t free_mem = 0, total_mem = 0;
                cudaMemGetInfo(&free_mem, &total_mem);
                size_t safety = 128 << 20;  // 128 MiB reserve
                size_t budget = (free_mem > safety) ? free_mem - safety : 0;
                budget = static_cast<size_t>(budget * 0.15);  // 15% of available
                if (expert_cache_.init(max_expert_raw, budget, vram_alloc_)) {
                    IMP_LOG_INFO("Expert LRU cache: %d slots (%.2f MiB / %.2f MiB budget)",
                                 expert_cache_.n_slots_,
                                 expert_cache_.n_slots_ * max_expert_raw / (1024.0 * 1024.0),
                                 budget / (1024.0 * 1024.0));
                }
            }
        }

        // Batch dequant buffer: sized for a chunk of experts (L2-resident strategy).
        // We dequant a chunk of experts to FP16, then immediately GEMM while the
        // FP16 data is still warm in L2 cache (~96 MB on RTX 5090). This avoids
        // writing the FP16 intermediate to DRAM entirely, saving ~5x DRAM traffic.
        // Skip allocation if experts are on host (batch dequant only useful for on-device experts).
        if (!skip_batch_dequant) {
            int targets[] = {cfg.n_experts, cfg.n_experts / 2, 32, 16};
            bool allocated = false;
            for (int ne_try : targets) {
                if (ne_try <= 0) continue;
                ne_try = std::min(ne_try, cfg.n_experts);
                size_t sz = static_cast<size_t>(ne_try) * eff * d * sizeof(half);
                moe_.batch_dequant_buf = vram_alloc(vram_alloc_, sz, "moe_batch_dequant");
                if (!moe_.batch_dequant_buf) {
                    IMP_LOG_DEBUG("MoE dequant buf alloc failed for %d experts", ne_try);
                    continue;
                }
                moe_.batch_dequant_buf_size = sz;
                allocated = true;
                IMP_LOG_INFO("MoE batch dequant buffer: %.2f MiB (%d experts)",
                             sz / (1024.0 * 1024.0), ne_try);
                break;
            }
            if (!allocated) {
                IMP_LOG_INFO("MoE batch dequant buffer: skipped (VRAM insufficient)");
                moe_.batch_dequant_buf = nullptr;
                moe_.batch_dequant_buf_size = 0;
            }
        } else {
            IMP_LOG_INFO("MoE batch dequant buffer: skipped (experts on host)");
            moe_.batch_dequant_buf = nullptr;
            moe_.batch_dequant_buf_size = 0;
        }

        // Pre-allocated device pointer arrays for batched MoE GEMM.
        // 3 arrays × n_experts void pointers = trivial memory (< 4 KB).
        // Eliminates cudaMallocAsync/FreeAsync from the hot path.
        if (cfg.n_experts > 0) {
            size_t ptr_bytes = 3 * static_cast<size_t>(cfg.n_experts) * sizeof(void*);
            cudaError_t err = cudaMalloc(&moe_.d_work_ptrs, ptr_bytes);
            if (err == cudaSuccess) {
                moe_.d_work_ptrs_count = cfg.n_experts;
            } else {
                IMP_LOG_DEBUG("Optional MoE work ptrs alloc failed: %s", cudaGetErrorString(err));
                moe_.d_work_ptrs = nullptr;
                moe_.d_work_ptrs_count = 0;
            }

            // Per-expert FP8 scale buffer (trivial: 128 experts × 4 bytes = 512 bytes).
            size_t scale_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(float);
            err = cudaMalloc(&moe_.d_fp8_scales, scale_bytes);
            if (err != cudaSuccess) {
                IMP_LOG_DEBUG("Optional MoE FP8 scales alloc failed: %s", cudaGetErrorString(err));
                moe_.d_fp8_scales = nullptr;
            }

            // Device-side weight pointer array for device-grouped GEMM.
            size_t wptr_bytes = static_cast<size_t>(cfg.n_experts) * sizeof(void*);
            err = cudaMalloc(&moe_.d_weight_ptrs, wptr_bytes);
            if (err == cudaSuccess) {
                moe_.d_weight_ptrs_count = cfg.n_experts;
            } else {
                IMP_LOG_DEBUG("Optional MoE weight ptrs alloc failed: %s", cudaGetErrorString(err));
                moe_.d_weight_ptrs = nullptr;
                moe_.d_weight_ptrs_count = 0;
            }
        }
    }

    // FP8 activation scratch buffers (for FP8 prefill weight cache)
    if (wcache_.use_fp8) {
        int max_dim = cfg.d_model;
        if (cfg.d_ff > 0) max_dim = std::max(max_dim, cfg.d_ff);
        max_dim = std::max(max_dim, cfg.n_heads * (cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / cfg.n_heads)));
        // SSM dimensions
        if (cfg.ssm_inner_size > 0) {
            int conv_ch = cfg.ssm_inner_size + 2 * cfg.ssm_group_count * cfg.ssm_state_size;
            int ssm_in_dim = cfg.ssm_inner_size + conv_ch + cfg.ssm_dt_rank;
            max_dim = std::max(max_dim, ssm_in_dim);
            max_dim = std::max(max_dim, cfg.ssm_inner_size);
        }
        qscratch_.fp8_act_size = static_cast<size_t>(max_tokens_) * max_dim;
        qscratch_.fp8_act = vram_alloc(vram_alloc_, qscratch_.fp8_act_size, "fp8_activation");
        if (!qscratch_.fp8_act) {
            IMP_LOG_WARN("Failed to allocate FP8 activation buffer (%.1f MiB)",
                         qscratch_.fp8_act_size / (1024.0 * 1024.0));
            qscratch_.fp8_act_size = 0;
        }
        {
            cudaError_t serr = cudaMalloc(reinterpret_cast<void**>(&qscratch_.d_act_scale), sizeof(float));
            if (serr != cudaSuccess) {
                IMP_LOG_WARN("Failed to allocate FP8 act scale: %s", cudaGetErrorString(serr));
                qscratch_.d_act_scale = nullptr;
            }
        }
        // Pre-allocate reduction buffers for async FP8 activation quantization.
        // Eliminates per-call cudaMalloc + cudaStreamSynchronize from the hot path.
        if (qscratch_.fp8_act && qscratch_.d_act_scale) {
            int max_n = static_cast<int>(qscratch_.fp8_act_size);  // max elements
            int threads_needed = (max_n + 3) / 4;  // kElemsPerThread=4
            qscratch_.fp8_max_grid = (threads_needed + 255) / 256;  // kBlockSize=256
            cudaError_t e1 = cudaMalloc(&qscratch_.d_fp8_block_maxes, static_cast<size_t>(qscratch_.fp8_max_grid) * sizeof(float));
            cudaError_t e2 = cudaMalloc(&qscratch_.d_fp8_absmax, sizeof(float));
            if (e1 != cudaSuccess || e2 != cudaSuccess || !qscratch_.d_fp8_block_maxes || !qscratch_.d_fp8_absmax) {
                IMP_LOG_WARN("Failed to allocate FP8 reduction buffers — will use sync path");
                if (qscratch_.d_fp8_block_maxes) { cudaFree(qscratch_.d_fp8_block_maxes); qscratch_.d_fp8_block_maxes = nullptr; }
                if (qscratch_.d_fp8_absmax) { cudaFree(qscratch_.d_fp8_absmax); qscratch_.d_fp8_absmax = nullptr; }
                qscratch_.fp8_max_grid = 0;
            }
            IMP_LOG_INFO("FP8 activation scratch: %.2f MiB (max_tokens=%d, max_dim=%d, async reduction grid=%d)",
                         qscratch_.fp8_act_size / (1024.0 * 1024.0), max_tokens_, max_dim, qscratch_.fp8_max_grid);
        }
    }

    // CUTLASS sm_120 NVFP4 activation buffers: pre-allocate for max prefill dimensions.
    // Only needed when NVFP4 decode is active and sm_120 is available.
    if (wcache_.nvfp4_decode_mode > 0 && cutlass_sm120_nvfp4_available()) {
        int max_k = 0;
        int max_n = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data && w->ndim >= 2) {
                    max_n = std::max(max_n, static_cast<int>(w->shape[0]));
                    max_k = std::max(max_k, static_cast<int>(w->shape[1]));
                }
            }
        }
        if (max_k > 0) {
            // Activation packed data: [max_tokens, max_K/2]
            qscratch_.cutlass_act_data_size = static_cast<size_t>(max_tokens_) * max_k / 2;
            // SfAtom scale factors for activation
            qscratch_.cutlass_act_sf_size = cutlass_nvfp4_sf_size(max_tokens_, max_k);
            // CUTLASS GEMM workspace
            qscratch_.cutlass_workspace_size = gemm_nvfp4_cutlass_sm120_workspace(max_tokens_, max_n, max_k);

            qscratch_.cutlass_act_data = vram_alloc(vram_alloc_, qscratch_.cutlass_act_data_size, "cutlass_act_data");
            qscratch_.cutlass_act_sf = vram_alloc(vram_alloc_, qscratch_.cutlass_act_sf_size, "cutlass_act_sf");
            qscratch_.cutlass_workspace = (qscratch_.cutlass_workspace_size > 0)
                               ? vram_alloc(vram_alloc_, qscratch_.cutlass_workspace_size, "cutlass_workspace")
                               : nullptr;
            if (!qscratch_.cutlass_act_data || !qscratch_.cutlass_act_sf ||
                (qscratch_.cutlass_workspace_size > 0 && !qscratch_.cutlass_workspace)) {
                IMP_LOG_WARN("Failed to allocate CUTLASS NVFP4 activation buffers, native FP4 prefill disabled");
                if (qscratch_.cutlass_act_data) { vram_free(vram_alloc_, qscratch_.cutlass_act_data); qscratch_.cutlass_act_data = nullptr; }
                if (qscratch_.cutlass_act_sf) { vram_free(vram_alloc_, qscratch_.cutlass_act_sf); qscratch_.cutlass_act_sf = nullptr; }
                if (qscratch_.cutlass_workspace) { vram_free(vram_alloc_, qscratch_.cutlass_workspace); qscratch_.cutlass_workspace = nullptr; }
                qscratch_.cutlass_act_data_size = 0;
                qscratch_.cutlass_act_sf_size = 0;
                qscratch_.cutlass_workspace_size = 0;
            } else {
                IMP_LOG_INFO("CUTLASS NVFP4 activation scratch: %.2f MiB (data=%.2f, sf=%.2f, ws=%.2f)",
                             (qscratch_.cutlass_act_data_size + qscratch_.cutlass_act_sf_size + qscratch_.cutlass_workspace_size) / (1024.0 * 1024.0),
                             qscratch_.cutlass_act_data_size / (1024.0 * 1024.0),
                             qscratch_.cutlass_act_sf_size / (1024.0 * 1024.0),
                             qscratch_.cutlass_workspace_size / (1024.0 * 1024.0));

                // MXFP4 activation buffers: shares packed data with NVFP4, only needs
                // separate UE8M0 scale factors (SFVecSize=32 vs NVFP4's 16).
                if (cutlass_sm120_mxfp4_available()) {
                    qscratch_.mxfp4_act_sf_size = cutlass_mxfp4_sf_size(max_tokens_, max_k);
                    qscratch_.mxfp4_workspace_size = gemm_mxfp4_cutlass_sm120_workspace(max_tokens_, max_n, max_k);
                    qscratch_.mxfp4_act_sf = vram_alloc(vram_alloc_, qscratch_.mxfp4_act_sf_size, "mxfp4_act_sf");
                    qscratch_.mxfp4_workspace = (qscratch_.mxfp4_workspace_size > 0)
                                     ? vram_alloc(vram_alloc_, qscratch_.mxfp4_workspace_size, "mxfp4_workspace")
                                     : nullptr;
                    if (!qscratch_.mxfp4_act_sf ||
                        (qscratch_.mxfp4_workspace_size > 0 && !qscratch_.mxfp4_workspace)) {
                        IMP_LOG_WARN("Failed to allocate MXFP4 activation buffers, MXFP4 prefill disabled");
                        if (qscratch_.mxfp4_act_sf) { vram_free(vram_alloc_, qscratch_.mxfp4_act_sf); qscratch_.mxfp4_act_sf = nullptr; }
                        if (qscratch_.mxfp4_workspace) { vram_free(vram_alloc_, qscratch_.mxfp4_workspace); qscratch_.mxfp4_workspace = nullptr; }
                        qscratch_.mxfp4_act_sf_size = 0;
                        qscratch_.mxfp4_workspace_size = 0;
                    } else {
                        IMP_LOG_INFO("CUTLASS MXFP4 activation scratch: sf=%.2f MiB, ws=%.2f MiB",
                                     qscratch_.mxfp4_act_sf_size / (1024.0 * 1024.0),
                                     qscratch_.mxfp4_workspace_size / (1024.0 * 1024.0));
                    }
                }
            }
        }
    }
}

void GraphExecutor::release_moe_batch_buf() {
    if (moe_.batch_dequant_buf) {
        size_t freed = moe_.batch_dequant_buf_size;
        vram_free(vram_alloc_, moe_.batch_dequant_buf);
        moe_.batch_dequant_buf = nullptr;
        moe_.batch_dequant_buf_size = 0;
        IMP_LOG_INFO("Released MoE batch dequant buffer: %.2f MiB (experts on host)",
                     freed / (1024.0 * 1024.0));
    }
}

void GraphExecutor::free_buffers() {
    // Helper: free through VRAMAllocator if pointer was tracked, else cudaFree.
    auto vfree = [this](void*& p) {
        if (p) { vram_free(vram_alloc_, p); p = nullptr; }
    };

    // Free TurboQuant QJL projection
    qjl_destroy(qjl_proj_);

    // Free LongRoPE frequency tables
    if (longrope_short_freqs_) { cudaFree(longrope_short_freqs_); longrope_short_freqs_ = nullptr; }
    if (longrope_long_freqs_)  { cudaFree(longrope_long_freqs_);  longrope_long_freqs_  = nullptr; }
    longrope_n_pairs_ = 0;
    longrope_orig_max_pos_ = 0;

    // Free all weight caches (FP16, FP8, NVFP4, CUTLASS, fused KV/gate+up, migrated/overflow)
    wcache_.free(vram_alloc_);

    qscratch_.free(vram_alloc_);

    moe_.free(vram_alloc_);
    expert_cache_.destroy();
    if (d_sample_result_) {
        cudaFree(d_sample_result_);
        d_sample_result_ = nullptr;
    }
    if (h_sample_pinned_) {
        cudaFreeHost(h_sample_pinned_);
        h_sample_pinned_ = nullptr;
    }
    if (h_logits_pinned_) {
        cudaFreeHost(h_logits_pinned_);
        h_logits_pinned_ = nullptr;
        h_logits_pinned_size_ = 0;
    }
    vfree(attn_scores_buf_);
    attn_scores_buf_size_ = 0;
#ifdef IMP_USE_CUTLASS
    cutlass_fmha_free_workspace();
#endif
    vfree(shared_workspace_);
    shared_workspace_size_ = 0;
    vfree(persistent_workspace_);
    persistent_workspace_size_ = 0;
    vfree(fp32_accum_buf_);
    ssm_layer_map_.clear();
    initialized_ = false;
}

// ---------------------------------------------------------------------------
// Pre-dequantize quantized weights to FP16 on GPU
// ---------------------------------------------------------------------------

void GraphExecutor::pre_dequant_weights(cudaStream_t stream, const VRAMBudget& budget) {
    if (!initialized_ || !model_) return;

    const auto& cfg = model_->config();
    size_t total_cache_bytes = 0;
    int cached_count = 0;
    bool budget_exhausted = false;

    // Compute effective cache budget from free VRAM minus reserve.
    // This preserves the existing per-phase budget tracking while the VRAMBudget
    // struct controls strategy-level decisions (which phases to skip).
    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);
    // Reserve at least 10% of total VRAM as headroom to avoid shared/system
    // memory fallback on WSL2 (not visible via nvidia-smi).
    size_t min_reserve = std::max(budget.reserve_bytes, total_vram / 10);
    size_t remaining_budget = (free_vram > min_reserve)
                              ? (free_vram - min_reserve) : 0;

    // Helper: does this qtype benefit from NVFP4 conversion? (> 4.5 bits/elem)
    auto nvfp4_beneficial = [](GGMLQuantType qt) -> bool {
        switch (qt) {
            case GGMLQuantType::Q8_0: case GGMLQuantType::Q8_K:
            case GGMLQuantType::Q6_K: case GGMLQuantType::Q5_K:
                return true;
            default: return false;
        }
    };

    if (wcache_.use_fp8) {
        // Skip Phase 1 entirely: FP8 cache (Phase 2) is the primary path.
        // FP8 is 50% smaller than FP16 and uses FP8×FP8 cuBLASLt (2x throughput
        // on sm_120 tensor cores).  Fused KV/gate+up (saving 1 launch each) are
        // replaced by individual FP8 GEMMs with 2x throughput — net win.
        IMP_LOG_INFO("FP8 prefill: skipping FP16 cache (Phase 1), "
                     "all dense weights → FP8 cache (Phase 2)");
    } else if (budget.strategy == VRAMBudget::NVFP4_DECODE_ONLY) {
        // Skip Phase 1: sub-8-bit weights don't benefit from FP16 expansion.
        // NVFP4 decode cache is the priority — all VRAM goes to Phase 3.
        // Prefill uses CUTLASS NVFP4 GEMM (for eligible weights) or on-the-fly dequant.
        IMP_LOG_INFO("NVFP4 decode only: skipping FP16 cache (Phase 1), "
                     "VRAM reserved for NVFP4 decode cache");
    } else {
        // --- Phase 1: FP16 weight cache + fused KV + fused gate+up ---
        auto cache_weight = [&](const Tensor& w, GGMLQuantType qtype) {
            if (!w.data || !dequant_gpu_supported(qtype)) return;
            if (wcache_.fp16.count(w.data)) return;  // already cached
            if (budget_exhausted) return;

            int rows = static_cast<int>(w.shape[0]);
            int cols = static_cast<int>(w.shape[1]);
            size_t fp16_bytes = static_cast<size_t>(rows) * cols * sizeof(half);

            if (total_cache_bytes + fp16_bytes > remaining_budget) {
                budget_exhausted = true;
                IMP_LOG_INFO("FP16 cache: VRAM budget reached after %d tensors (%.1f / %.1f MiB), "
                             "remaining weights will use on-the-fly dequant",
                             cached_count, total_cache_bytes / (1024.0 * 1024.0),
                             remaining_budget / (1024.0 * 1024.0));
                return;
            }

            void* fp16_buf = vram_alloc(vram_alloc_, fp16_bytes, "fp16_weight_cache");
            if (!fp16_buf) {
                budget_exhausted = true;
                IMP_LOG_WARN("FP16 cache: allocation failed after %d tensors (%.1f MiB)",
                             cached_count, total_cache_bytes / (1024.0 * 1024.0));
                return;
            }

            dequant_gpu(w.data, fp16_buf, qtype, rows, cols, stream);

            Tensor fp16_tensor(fp16_buf, DType::FP16, w.ndim, w.shape, true);
            wcache_.fp16[w.data] = fp16_tensor;
            total_cache_bytes += fp16_bytes;
            cached_count++;
        };

        // Priority order: attention weights first (critical for cuBLAS prefill),
        // then SSM, shared experts, and dense FFN.  This ensures hybrid models
        // like Nemotron (23 SSM + 6 attention layers) cache all attention weights
        // before SSM weights exhaust the VRAM budget.
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            cache_weight(L.wq, L.wq_qtype);
            cache_weight(L.wk, L.wk_qtype);
            cache_weight(L.wv, L.wv_qtype);
            cache_weight(L.wo, L.wo_qtype);
        }
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            cache_weight(L.ssm_in, L.ssm_in_qtype);
            cache_weight(L.ssm_out, L.ssm_out_qtype);
            cache_weight(L.w_gate_shared, L.w_gate_shared_qtype);
            cache_weight(L.w_up_shared, L.w_up_shared_qtype);
            cache_weight(L.w_down_shared, L.w_down_shared_qtype);
            // When NVFP4 decode is active, skip dense FFN FP16 cache for eligible
            // weights.  Decode benefits more from NVFP4 (~47% BW reduction) than
            // prefill loses from on-the-fly dequant.  NVFP4 is also ~3.5x smaller
            // per tensor, so skipping FFN FP16 frees massive VRAM for full NVFP4.
            if (wcache_.nvfp4_decode_mode == 0 || !nvfp4_beneficial(L.w_gate_qtype))
                cache_weight(L.w_gate, L.w_gate_qtype);
            if (wcache_.nvfp4_decode_mode == 0 || !nvfp4_beneficial(L.w_up_qtype))
                cache_weight(L.w_up, L.w_up_qtype);
            if (wcache_.nvfp4_decode_mode == 0 || !nvfp4_beneficial(L.w_down_qtype))
                cache_weight(L.w_down, L.w_down_qtype);
        }

        // Create fused KV weights for strided batched prefill GEMM.
        // Each entry concatenates [wk; wv] as [2*nkv*hd, d_model] FP16 for one layer.
        int fused_kv_count = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            bool stop = false;
            if (create_fused_weight_pair(L.wk, L.wv, wcache_.fp16, vram_alloc_,
                                         total_cache_bytes, remaining_budget,
                                         stream, wcache_.fused_kv, i, stop))
                fused_kv_count++;
            else if (stop) break;
        }

        // Create fused gate+up weights for strided batched prefill GEMM.
        // Each entry concatenates [w_gate; w_up] as [2*d_ff, d_model] FP16 for one layer.
        int fused_gu_count = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            // Both must be the same shape (d_ff x d_model)
            if (L.w_gate.data && L.w_up.data &&
                (L.w_gate.shape[0] != L.w_up.shape[0] ||
                 L.w_gate.shape[1] != L.w_up.shape[1])) continue;
            bool stop = false;
            if (create_fused_weight_pair(L.w_gate, L.w_up, wcache_.fp16, vram_alloc_,
                                         total_cache_bytes, remaining_budget,
                                         stream, wcache_.fused_gate_up, i, stop))
                fused_gu_count++;
            else if (stop) break;
        }

        if (cached_count > 0) {
            cudaStreamSynchronize(stream);
            wcache_.fp16_bytes = total_cache_bytes;
            IMP_LOG_INFO("FP16 weight cache: %d tensors, %.2f MiB (incl. %d fused KV, %d fused gate+up)",
                         cached_count, total_cache_bytes / (1024.0 * 1024.0),
                         fused_kv_count, fused_gu_count);
        }
    } // end Phase 1

    // Deduct Phase 1 allocation from shared budget
    deduct_budget(remaining_budget, total_cache_bytes);

    // --- Phase 2: FP8 cache for uncached weights (primary when wcache_.use_fp8) ---
    // When wcache_.use_fp8 is true and Phase 1 was skipped, this is the primary path
    // for ALL dense projection weights.  FP8 is 50% smaller than FP16 and uses
    // FP8×FP8 cuBLASLt with 2x tensor core throughput on sm_120.
    // Uses qscratch_.dequant as FP16 staging buffer (stream ordering ensures safety).
    if (wcache_.use_fp8) {
        size_t fp8_total = 0;
        int fp8_count = 0;
        bool fp8_exhausted = false;

        // Collect weights to convert
        struct FP8OverflowEntry {
            const void* orig_ptr;
            Tensor weight;
            GGMLQuantType qtype;
            size_t n_elems;
        };
        std::vector<FP8OverflowEntry> fp8_entries;

        auto collect_weight_fp8 = [&](const Tensor& w, GGMLQuantType qtype) {
            if (!w.data || !dequant_gpu_supported(qtype)) return;
            if (wcache_.fp16.count(w.data)) return;
            if (wcache_.fp8.count(w.data)) return;
            if (fp8_exhausted) return;

            size_t n_elems = static_cast<size_t>(w.shape[0]) * w.shape[1];
            size_t fp8_bytes = n_elems;

            if (fp8_total + fp8_bytes + sizeof(float) > remaining_budget) {
                fp8_exhausted = true;
                IMP_LOG_INFO("FP8 overflow: budget reached after %d tensors (%.1f / %.1f MiB)",
                             fp8_count, fp8_total / (1024.0 * 1024.0),
                             remaining_budget / (1024.0 * 1024.0));
                return;
            }

            fp8_entries.push_back({w.data, w, qtype, n_elems});
            fp8_total += fp8_bytes + sizeof(float);
            fp8_count++;
        };

        // Same priority order — attention first, then SSM/FFN
        for_each_dense_weight(*model_, cfg, collect_weight_fp8);

        if (!fp8_entries.empty() && qscratch_.dequant) {
            // Pre-allocate reusable calibration temp buffers
            int max_grid = 0;
            size_t total_fp8_bytes = 0;
            for (auto& e : fp8_entries) {
                int threads_needed = (static_cast<int>(e.n_elems) + 3) / 4;
                int grid = (threads_needed + 255) / 256;
                if (grid > max_grid) max_grid = grid;
                total_fp8_bytes += e.n_elems;
            }

            float* d_block_maxes = nullptr;
            float* d_absmax = nullptr;
            float* d_scales_all = nullptr;
            cudaMalloc(&d_block_maxes, (size_t)max_grid * sizeof(float));
            cudaMalloc(&d_absmax, sizeof(float));
            cudaMalloc(&d_scales_all, fp8_entries.size() * sizeof(float));

            // Bulk-allocate all FP8 data
            uint8_t* d_fp8_bulk = static_cast<uint8_t*>(
                vram_alloc(vram_alloc_, total_fp8_bytes, "fp8_weight_cache"));
            if (!d_fp8_bulk) {
                cudaError_t e = cudaGetLastError();
                IMP_LOG_WARN("FP8 weight cache bulk alloc failed (%.1f MiB): %s",
                             total_fp8_bytes / (1024.0 * 1024.0), cudaGetErrorString(e));
            }

            int actual_count = 0;
            size_t fp8_offset = 0;
            for (size_t i = 0; i < fp8_entries.size() && d_fp8_bulk; i++) {
                auto& e = fp8_entries[i];
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);

                // Dequant to qscratch_.dequant (reused each iteration, stream-ordered)
                dequant_gpu(e.weight.data, qscratch_.dequant, e.qtype, rows, cols, stream);

                void* fp8_buf = d_fp8_bulk + fp8_offset;
                fp8_offset += e.n_elems;

                // Async calibrate + quantize (no host sync)
                calibrate_and_quantize_fp8_async(
                    qscratch_.dequant, fp8_buf, static_cast<int>(e.n_elems),
                    d_block_maxes, max_grid,
                    d_absmax, d_scales_all + static_cast<ptrdiff_t>(i), stream);

                Tensor fp8_t(fp8_buf, DType::FP8_E4M3, e.weight.ndim, e.weight.shape, true);
                wcache_.fp8[e.orig_ptr] = {fp8_t, 0.0f, d_scales_all + static_cast<ptrdiff_t>(i)};
                actual_count++;
            }

            if (actual_count > 0) {
                cudaStreamSynchronize(stream);
                // Read back scales
                std::vector<float> h_scales(actual_count);
                cudaMemcpy(h_scales.data(), d_scales_all, actual_count * sizeof(float),
                           cudaMemcpyDeviceToHost);
                for (int i = 0; i < actual_count; i++) {
                    auto it = wcache_.fp8.find(fp8_entries[i].orig_ptr);
                    if (it != wcache_.fp8.end()) {
                        it->second.host_scale = h_scales[i];
                    }
                }
            }

            cudaFree(d_block_maxes);
            cudaFree(d_absmax);
            // Track bulk buffers for cleanup
            wcache_.fp8_overflow_scales = d_scales_all;
            wcache_.fp8_overflow_count = actual_count;
            wcache_.fp8_overflow_data = d_fp8_bulk;
            wcache_.fp8_overflow_data_size = total_fp8_bytes;
            fp8_count = actual_count;
        }

        if (fp8_count > 0) {
            wcache_.fp8_bytes = fp8_total;
            size_t fp16_equivalent = 0;
            for (auto& [ptr, entry] : wcache_.fp8) {
                fp16_equivalent += entry.weight.numel() * sizeof(half);
            }
            IMP_LOG_INFO("FP8 weight cache: %d tensors, %.2f MiB (%.2f MiB saved vs FP16)",
                         fp8_count, fp8_total / (1024.0 * 1024.0),
                         (fp16_equivalent - fp8_total) / (1024.0 * 1024.0));
        } else {
            IMP_LOG_INFO("FP8 prefill: no weights cached (budget=0 or no eligible weights)");
        }
    }

    // Deduct Phase 2 allocation from shared budget
    deduct_budget(remaining_budget, wcache_.fp8_bytes);

    // --- Phase 3: NVFP4 decode weight cache ---
    // Converts eligible weights (> 4.5 bits/elem) to NVFP4 format for faster
    // decode GEMV.  Mode 2 ("only") uses incremental processing: quantize from
    // FP16 cache and free each entry immediately (NVFP4 ≈ 28% of FP16 size, so
    // each conversion is net VRAM-negative, bootstrapping space for more tensors).
    // Mode 1 ("additive") uses standard batch processing with FP16 cache intact.
    if (wcache_.nvfp4_decode_mode > 0) {
        const char* mode_str = (wcache_.nvfp4_decode_mode == 1) ? "additive" : "only";

        // Collect eligible weights first, then process.
        struct NvFP4Entry {
            const void* orig_ptr;
            Tensor weight;
            GGMLQuantType qtype;
            bool from_scratch;
        };
        std::vector<NvFP4Entry> nvfp4_entries;

        auto collect_weight_nvfp4 = [&](const Tensor& w, GGMLQuantType qtype) {
            if (!w.data) return;
            if (!nvfp4_beneficial(qtype)) return;
            if (wcache_.nvfp4.count(w.data)) return;

            int cols = static_cast<int>(w.shape[1]);
            if (cols % 16 != 0) return;

            bool from_scratch = (wcache_.fp16.find(w.data) == wcache_.fp16.end());
            if (from_scratch && (!dequant_gpu_supported(qtype) || !qscratch_.dequant)) return;
            nvfp4_entries.push_back({w.data, w, qtype, from_scratch});
        };

        // LM head first: largest single weight (vocab × d_model), biggest bandwidth win.
        collect_weight_nvfp4(model_->output_proj(), model_->out_proj_qtype_);

        // Dense attention + FFN: every tensor benefits every decode step.
        for_each_dense_weight(*model_, cfg, collect_weight_nvfp4);

        if (wcache_.nvfp4_decode_mode == 2 && !nvfp4_entries.empty()) {
            // Mode 2 incremental: process FP16-cached entries first (each conversion
            // frees net VRAM since NVFP4 ≈ 28% of FP16), then from-scratch entries.
            // Sort: FP16-cached first (smallest first to bootstrap), then from-scratch.
            std::stable_sort(nvfp4_entries.begin(), nvfp4_entries.end(),
                [](const NvFP4Entry& a, const NvFP4Entry& b) {
                    if (a.from_scratch != b.from_scratch) return !a.from_scratch;
                    size_t a_sz = static_cast<size_t>(a.weight.shape[0]) * a.weight.shape[1];
                    size_t b_sz = static_cast<size_t>(b.weight.shape[0]) * b.weight.shape[1];
                    return a_sz < b_sz;
                });

            float* d_absmax_buf = nullptr;
            float* d_tscale_buf = nullptr;
            cudaMalloc(&d_absmax_buf, sizeof(float));
            cudaMalloc(&d_tscale_buf, sizeof(float));

            int actual_count = 0;
            size_t actual_bytes = 0;
            int actual_from_fp16 = 0;
            int actual_from_scratch = 0;

            for (auto& e : nvfp4_entries) {
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);
                size_t nvfp4_bytes = static_cast<size_t>(rows) * cols / 2 +
                                     static_cast<size_t>(rows) * cols / 16 + 4;

                // Check actual free VRAM (10% of total as safety margin)
                size_t free_mem = 0, total_mem = 0;
                cudaMemGetInfo(&free_mem, &total_mem);
                size_t nvfp4_safety = std::max(total_mem / 10, static_cast<size_t>(1024 * 1024));
                if (free_mem < nvfp4_bytes + nvfp4_safety) {
                    IMP_LOG_INFO("NVFP4 incremental: VRAM exhausted after %d tensors "
                                 "(%.1f MiB, %.1f MiB free)", actual_count,
                                 actual_bytes / (1024.0 * 1024.0), free_mem / (1024.0 * 1024.0));
                    break;
                }

                const half* fp16_ptr = nullptr;
                void* tmp_buf = nullptr;

                if (e.from_scratch) {
                    size_t need = static_cast<size_t>(rows) * cols * sizeof(half);
                    void* dq_buf = qscratch_.dequant;
                    if (need > qscratch_.dequant_size) {
                        if (cudaMalloc(&tmp_buf, need) != cudaSuccess || !tmp_buf) continue;
                        dq_buf = tmp_buf;
                    }
                    dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);
                    fp16_ptr = reinterpret_cast<const half*>(dq_buf);
                } else {
                    auto it = wcache_.fp16.find(e.orig_ptr);
                    fp16_ptr = reinterpret_cast<const half*>(it->second.data);
                }

                Tensor fp16_view(const_cast<half*>(fp16_ptr), DType::FP16, 2,
                                 e.weight.shape, true);

                NvFP4QuantResult result;
                quantize_fp16_to_nvfp4_async(fp16_view, result,
                                              d_absmax_buf, d_tscale_buf, stream);

                // Sync immediately so we can read tensor_scale and free FP16
                cudaStreamSynchronize(stream);

                float h_tscale;
                cudaMemcpy(&h_tscale, d_tscale_buf, sizeof(float), cudaMemcpyDeviceToHost);
                result.tensor_scale = h_tscale;
                wcache_.nvfp4[e.orig_ptr] = result;
                actual_bytes += nvfp4_bytes;
                actual_count++;

                if (tmp_buf) cudaFree(tmp_buf);

                // Free FP16 cache entry to reclaim VRAM for next weight
                if (!e.from_scratch) {
                    auto it = wcache_.fp16.find(e.orig_ptr);
                    if (it != wcache_.fp16.end()) {
                        size_t freed = it->second.nbytes();
                        vram_free(vram_alloc_, it->second.data);
                        wcache_.fp16.erase(it);
                        wcache_.fp16_bytes -= freed;
                        actual_from_fp16++;
                    }
                } else {
                    actual_from_scratch++;
                }
            }

            cudaFree(d_absmax_buf);
            cudaFree(d_tscale_buf);

            wcache_.nvfp4_bytes = actual_bytes;
            IMP_LOG_INFO("NVFP4 decode cache: %d tensors, %.2f MiB "
                         "(%d from FP16, %d from scratch, mode: %s)",
                         actual_count, actual_bytes / (1024.0 * 1024.0),
                         actual_from_fp16, actual_from_scratch, mode_str);
        } else if (!nvfp4_entries.empty()) {
            // Mode 1 standard batch: quantize entries that fit in budget, single sync.
            size_t budget_used = 0;
            int nvfp4_count = 0;
            int nvfp4_from_scratch = 0;
            bool budget_exhausted = false;

            std::vector<NvFP4Entry> budgeted;
            for (auto& e : nvfp4_entries) {
                size_t rows = e.weight.shape[0], cols = e.weight.shape[1];
                size_t nvfp4_bytes = rows * cols / 2 + rows * cols / 16 + 4;
                if (budget_used + nvfp4_bytes > remaining_budget) {
                    if (!budget_exhausted) {
                        budget_exhausted = true;
                        IMP_LOG_INFO("NVFP4 cache: VRAM budget reached after %d/%zu tensors "
                                     "(%.1f / %.1f MiB)",
                                     nvfp4_count, nvfp4_entries.size(),
                                     budget_used / (1024.0 * 1024.0),
                                     remaining_budget / (1024.0 * 1024.0));
                    }
                    continue;
                }
                budget_used += nvfp4_bytes;
                nvfp4_count++;
                if (e.from_scratch) nvfp4_from_scratch++;
                budgeted.push_back(e);
            }

            float* d_absmax_buf = nullptr;
            cudaMalloc(&d_absmax_buf, sizeof(float));

            float* d_tscales_all = nullptr;
            cudaMalloc(&d_tscales_all, budgeted.size() * sizeof(float));

            std::vector<void*> tmp_bufs;
            for (size_t i = 0; i < budgeted.size(); i++) {
                auto& e = budgeted[i];
                const half* fp16_ptr = nullptr;
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);

                if (e.from_scratch) {
                    size_t need = static_cast<size_t>(rows) * cols * sizeof(half);
                    void* dq_buf = qscratch_.dequant;
                    if (need > qscratch_.dequant_size) {
                        void* tmp = nullptr;
                        if (cudaMalloc(&tmp, need) != cudaSuccess || !tmp) continue;
                        dq_buf = tmp;
                        tmp_bufs.push_back(tmp);
                    }
                    dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);
                    fp16_ptr = reinterpret_cast<const half*>(dq_buf);
                } else {
                    auto it = wcache_.fp16.find(e.orig_ptr);
                    fp16_ptr = reinterpret_cast<const half*>(it->second.data);
                }

                Tensor fp16_view(const_cast<half*>(fp16_ptr), DType::FP16, 2,
                                 e.weight.shape, true);

                NvFP4QuantResult result;
                quantize_fp16_to_nvfp4_async(fp16_view, result,
                                              d_absmax_buf,
                                              d_tscales_all + i,
                                              stream);
                wcache_.nvfp4[e.orig_ptr] = result;
            }

            cudaStreamSynchronize(stream);
            for (void* p : tmp_bufs) cudaFree(p);

            std::vector<float> h_tscales(budgeted.size());
            cudaMemcpy(h_tscales.data(), d_tscales_all,
                       budgeted.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < budgeted.size(); i++) {
                auto it = wcache_.nvfp4.find(budgeted[i].orig_ptr);
                if (it != wcache_.nvfp4.end()) {
                    it->second.tensor_scale = h_tscales[i];
                }
            }

            cudaFree(d_absmax_buf);
            cudaFree(d_tscales_all);

            wcache_.nvfp4_bytes = budget_used;
            if (nvfp4_from_scratch > 0) {
                IMP_LOG_INFO("NVFP4 decode cache: %d tensors, %.2f MiB (%d from FP16 cache, %d via dequant scratch, mode: %s)",
                             nvfp4_count, budget_used / (1024.0 * 1024.0),
                             nvfp4_count - nvfp4_from_scratch, nvfp4_from_scratch, mode_str);
            } else {
                IMP_LOG_INFO("NVFP4 decode cache: %d tensors, %.2f MiB (mode: %s)",
                             nvfp4_count, budget_used / (1024.0 * 1024.0), mode_str);
            }
        }

        // In "only" mode (2), release remaining FP16 cache.
        // Before freeing, migrate FP16 weights to FP8 cache so prefill retains
        // fast FP8 GEMM.  FP8 = half the size of FP16, net 50% VRAM savings.
        if (wcache_.nvfp4_decode_mode == 2 && !wcache_.fp16.empty()) {
            int migrated = 0;
            size_t migrated_bytes = 0;
            if (wcache_.use_fp8) {
                struct MigrateEntry {
                    const void* orig_ptr;
                    Tensor fp16_tensor;
                    size_t n_elems;
                };
                std::vector<MigrateEntry> to_migrate;
                for (auto& [orig_ptr, fp16_tensor] : wcache_.fp16) {
                    if (wcache_.fp8.count(orig_ptr)) continue;
                    size_t n = static_cast<size_t>(fp16_tensor.shape[0]) * fp16_tensor.shape[1];
                    to_migrate.push_back({orig_ptr, fp16_tensor, n});
                }

                if (!to_migrate.empty()) {
                    int max_grid = 0;
                    size_t total_fp8_bytes = 0;
                    for (auto& e : to_migrate) {
                        int threads_needed = (static_cast<int>(e.n_elems) + 3) / 4;
                        int grid = (threads_needed + 255) / 256;
                        if (grid > max_grid) max_grid = grid;
                        total_fp8_bytes += e.n_elems;
                    }

                    float* d_block_maxes = nullptr;
                    float* d_absmax = nullptr;
                    cudaMalloc(&d_block_maxes, (size_t)max_grid * sizeof(float));
                    cudaMalloc(&d_absmax, sizeof(float));

                    float* d_scales_all = nullptr;
                    cudaMalloc(&d_scales_all, to_migrate.size() * sizeof(float));

                    uint8_t* d_fp8_bulk = nullptr;
                    d_fp8_bulk = static_cast<uint8_t*>(
                        vram_alloc(vram_alloc_, total_fp8_bytes, "fp8_migration_cache"));
                    if (!d_fp8_bulk) {
                        cudaError_t e = cudaGetLastError();
                        IMP_LOG_WARN("FP8 migration cache alloc failed (%.1f MiB): %s",
                                     total_fp8_bytes / (1024.0 * 1024.0), cudaGetErrorString(e));
                    }

                    size_t fp8_offset = 0;
                    for (size_t i = 0; i < to_migrate.size() && d_fp8_bulk; i++) {
                        auto& e = to_migrate[i];
                        void* fp8_buf = d_fp8_bulk + fp8_offset;
                        fp8_offset += e.n_elems;

                        calibrate_and_quantize_fp8_async(
                            e.fp16_tensor.data, fp8_buf, static_cast<int>(e.n_elems),
                            d_block_maxes, max_grid,
                            d_absmax, d_scales_all + i, stream);

                        Tensor fp8_t(fp8_buf, DType::FP8_E4M3, e.fp16_tensor.ndim,
                                     e.fp16_tensor.shape, true);
                        wcache_.fp8[e.orig_ptr] = {fp8_t, 0.0f, d_scales_all + static_cast<ptrdiff_t>(i)};
                        migrated++;
                        migrated_bytes += e.n_elems + sizeof(float);
                    }

                    wcache_.fp8_migrated_data = d_fp8_bulk;
                    wcache_.fp8_migrated_data_size = total_fp8_bytes;

                    if (migrated > 0) {
                        cudaStreamSynchronize(stream);
                        std::vector<float> h_scales(migrated);
                        cudaMemcpy(h_scales.data(), d_scales_all, migrated * sizeof(float),
                                   cudaMemcpyDeviceToHost);
                        int idx = 0;
                        for (size_t i = 0; i < to_migrate.size() && idx < migrated; i++, idx++) {
                            auto it = wcache_.fp8.find(to_migrate[i].orig_ptr);
                            if (it != wcache_.fp8.end()) {
                                it->second.host_scale = h_scales[idx];
                            }
                        }
                    }

                    cudaFree(d_block_maxes);
                    cudaFree(d_absmax);
                    wcache_.fp8_migrated_scales = d_scales_all;
                    wcache_.fp8_migrated_count = migrated;
                }
            }

            // Free remaining FP16 cache
            for (auto& [ptr, tensor] : wcache_.fp16) {
                vram_free(vram_alloc_, tensor.data);
            }
            size_t freed = wcache_.fp16_bytes;
            wcache_.fp16.clear();
            wcache_.fp16_bytes = 0;

            // Free fused caches (prefill uses individual FP8 weights)
            for (auto& [idx, tensor] : wcache_.fused_kv) {
                if (tensor.data) vram_free(vram_alloc_, tensor.data);
            }
            wcache_.fused_kv.clear();
            for (auto& [idx, tensor] : wcache_.fused_gate_up) {
                if (tensor.data) vram_free(vram_alloc_, tensor.data);
            }
            wcache_.fused_gate_up.clear();

            remaining_budget += freed;
            wcache_.fp8_bytes += migrated_bytes;
            IMP_LOG_INFO("NVFP4 only mode: freed FP16 cache (%.2f MiB), migrated %d weights to FP8 (%.2f MiB)",
                         freed / (1024.0 * 1024.0), migrated, migrated_bytes / (1024.0 * 1024.0));
        }

        // --- NVFP4 second pass: cache remaining tensors with freed VRAM ---
        // After FP16-Free and FP8 migration, VRAM that was locked by FP16 cache is
        // now available. Re-run NVFP4 for entries that were skipped due to VRAM pressure.
        if (budget.nvfp4_second_pass && !nvfp4_entries.empty()) {
            float* d_absmax_buf2 = nullptr;
            float* d_tscale_buf2 = nullptr;
            cudaMalloc(&d_absmax_buf2, sizeof(float));
            cudaMalloc(&d_tscale_buf2, sizeof(float));

            int second_count = 0;
            size_t second_bytes = 0;

            for (auto& e : nvfp4_entries) {
                if (wcache_.nvfp4.count(e.orig_ptr)) continue;  // already cached
                int rows = static_cast<int>(e.weight.shape[0]);
                int cols = static_cast<int>(e.weight.shape[1]);
                size_t nvfp4_bytes = static_cast<size_t>(rows) * cols / 2 +
                                     static_cast<size_t>(rows) * cols / 16 + 4;

                size_t free_mem2 = 0, total_mem2 = 0;
                cudaMemGetInfo(&free_mem2, &total_mem2);
                size_t nvfp4_safety2 = std::max(total_mem2 / 10, static_cast<size_t>(1024 * 1024));
                if (free_mem2 < nvfp4_bytes + nvfp4_safety2) break;

                // Dequant from quantized weights via scratch buffer
                size_t need = static_cast<size_t>(rows) * cols * sizeof(half);
                void* dq_buf = qscratch_.dequant;
                void* tmp_buf = nullptr;
                if (!dequant_gpu_supported(e.qtype) || !qscratch_.dequant) continue;
                if (need > qscratch_.dequant_size) {
                    if (cudaMalloc(&tmp_buf, need) != cudaSuccess || !tmp_buf) continue;
                    dq_buf = tmp_buf;
                }
                dequant_gpu(e.weight.data, dq_buf, e.qtype, rows, cols, stream);

                Tensor fp16_view(reinterpret_cast<half*>(dq_buf), DType::FP16, 2,
                                 e.weight.shape, true);
                NvFP4QuantResult result;
                quantize_fp16_to_nvfp4_async(fp16_view, result,
                                              d_absmax_buf2, d_tscale_buf2, stream);
                cudaStreamSynchronize(stream);

                float h_tscale;
                cudaMemcpy(&h_tscale, d_tscale_buf2, sizeof(float), cudaMemcpyDeviceToHost);
                result.tensor_scale = h_tscale;
                wcache_.nvfp4[e.orig_ptr] = result;
                second_bytes += nvfp4_bytes;
                second_count++;

                if (tmp_buf) cudaFree(tmp_buf);
            }

            cudaFree(d_absmax_buf2);
            cudaFree(d_tscale_buf2);

            if (second_count > 0) {
                wcache_.nvfp4_bytes += second_bytes;
                IMP_LOG_INFO("NVFP4 second pass: %d additional tensors, %.2f MiB",
                             second_count, second_bytes / (1024.0 * 1024.0));
            }
        }

        // --- Phase 3b: Convert NVFP4 weights to CUTLASS sm_120 block-scaled format ---
        // Must be AFTER FP16 free to avoid peak VRAM exceeding physical memory.
        // The CUTLASS cache is a full copy (repacked data + SfAtom scales), so it
        // approximately doubles the NVFP4 cache VRAM.  Budget-aware: stop if VRAM runs out.
        if (!wcache_.nvfp4.empty() && cutlass_sm120_nvfp4_available()) {
            // After incremental mode, remaining_budget is stale.  Use actual free VRAM.
            size_t ct_budget;
            if (wcache_.nvfp4_decode_mode == 2) {
                cudaStreamSynchronize(stream);
                size_t free_mem = 0, total_mem = 0;
                cudaMemGetInfo(&free_mem, &total_mem);
                size_t kCtReserve = std::max(total_mem / 10, static_cast<size_t>(256ULL * 1024 * 1024));
                ct_budget = (free_mem > kCtReserve) ? (free_mem - kCtReserve) : 0;
            } else {
                ct_budget = (remaining_budget > wcache_.nvfp4_bytes)
                            ? (remaining_budget - wcache_.nvfp4_bytes) : 0;
            }
            int ct_count = 0;
            size_t ct_total = 0;
            bool ct_exhausted = false;
            for (auto& [ptr, nvfp4] : wcache_.nvfp4) {
                if (ct_exhausted) break;
                // Estimate CUTLASS allocation (only scale factors — data is borrowed)
                size_t est = cutlass_nvfp4_sf_size(static_cast<int>(nvfp4.N),
                                                    static_cast<int>(nvfp4.K));
                if (ct_total + est > ct_budget) {
                    ct_exhausted = true;
                    IMP_LOG_INFO("CUTLASS NVFP4 cache: VRAM budget reached after %d tensors "
                                 "(%.1f / %.1f MiB)",
                                 ct_count, ct_total / (1024.0 * 1024.0),
                                 ct_budget / (1024.0 * 1024.0));
                    break;
                }
                CutlassNvFP4Weight cw;
                convert_nvfp4_to_cutlass(nvfp4, cw, stream);
                if (cw.data) {
                    wcache_.cutlass_nvfp4[ptr] = cw;
                    ct_total += cw.sf_bytes;
                    ct_count++;
                }
            }
            if (ct_count > 0) {
                cudaStreamSynchronize(stream);
                wcache_.cutlass_nvfp4_bytes = ct_total;
                deduct_budget(remaining_budget, ct_total + wcache_.nvfp4_bytes);
                IMP_LOG_INFO("CUTLASS sm_120 NVFP4 weight cache: %d tensors, %.2f MiB",
                             ct_count, ct_total / (1024.0 * 1024.0));
            }

            // Phase 3c-native: register MXFP4 GGUF weights directly in CUTLASS cache.
            // These bypass NVFP4 entirely — the GGUF data is unpacked into
            // separate E2M1 data + SfAtom UE8M0 scales on GPU.
            // For native MXFP4, allocate activation buffers if not already done.
            if (cutlass_sm120_mxfp4_available()) {
                // Check if any layer has MXFP4 weights
                bool has_mxfp4 = false;
                auto check_mxfp4 = [&](const Tensor&, GGMLQuantType qt) {
                    if (qt == GGMLQuantType::MXFP4) has_mxfp4 = true;
                };
                for (int i = 0; i < cfg.n_layers && !has_mxfp4; i++) {
                    const auto& L = model_->layer(i);
                    check_mxfp4(L.wq, L.wq_qtype);
                    check_mxfp4(L.wk, L.wk_qtype);
                    check_mxfp4(L.w_gate, L.w_gate_qtype);
                }

                // Allocate MXFP4 scratch if needed and not already allocated
                if (has_mxfp4 && !qscratch_.mxfp4_act_sf) {
                    int max_k = 0, max_n = 0;
                    for (int i = 0; i < cfg.n_layers; i++) {
                        const auto& L = model_->layer(i);
                        if (L.wq.data && L.wq.ndim >= 2) {
                            max_n = std::max(max_n, (int)L.wq.shape[0]);
                            max_k = std::max(max_k, (int)L.wq.shape[1]);
                        }
                        if (L.w_gate.data && L.w_gate.ndim >= 2) {
                            max_n = std::max(max_n, (int)L.w_gate.shape[0]);
                            max_k = std::max(max_k, (int)L.w_gate.shape[1]);
                        }
                        if (L.w_down.data && L.w_down.ndim >= 2) {
                            max_n = std::max(max_n, (int)L.w_down.shape[0]);
                            max_k = std::max(max_k, (int)L.w_down.shape[1]);
                        }
                    }
                    if (max_k > 0) {
                        qscratch_.mxfp4_act_sf_size = cutlass_mxfp4_sf_size(max_tokens_, max_k);
                        qscratch_.mxfp4_workspace_size = gemm_mxfp4_cutlass_sm120_workspace(max_tokens_, max_n, max_k);
                        qscratch_.mxfp4_act_sf = vram_alloc(vram_alloc_, qscratch_.mxfp4_act_sf_size, "mxfp4_act_sf");
                        qscratch_.mxfp4_workspace = (qscratch_.mxfp4_workspace_size > 0)
                            ? vram_alloc(vram_alloc_, qscratch_.mxfp4_workspace_size, "mxfp4_workspace")
                            : nullptr;
                        // Also need CUTLASS activation data buffer
                        if (!qscratch_.cutlass_act_data) {
                            qscratch_.cutlass_act_data_size = static_cast<size_t>(max_tokens_) * (max_k / 2);
                            qscratch_.cutlass_act_data = vram_alloc(vram_alloc_, qscratch_.cutlass_act_data_size, "cutlass_act_data");
                        }
                        IMP_LOG_INFO("Native MXFP4: allocated activation scratch (sf=%.2f MiB)",
                                     qscratch_.mxfp4_act_sf_size / (1024.0 * 1024.0));
                    }
                }
            }

            // Convert NVFP4 weights to MXFP4 (UE8M0 scales) if MXFP4 prefill is enabled.
            // Same packed FP4 data (borrowed), only allocates new scale factor buffers.
            // Note: Hadamard rotation requires MR-GPTQ pre-rotated weights (SafeTensors).
            // For GGUF models, we use direct scale conversion (no rotation).
            if (wcache_.use_mxfp4 && qscratch_.mxfp4_act_sf != nullptr && cutlass_sm120_mxfp4_available()) {
                int mx_count = 0;
                size_t mx_total = 0;
                for (auto& [ptr, nvfp4] : wcache_.nvfp4) {
                    // Only convert weights where K is multiple of 32 (MXFP4 requirement)
                    if (nvfp4.K % 32 != 0) continue;
                    CutlassMxFP4Weight mw;
                    convert_nvfp4_to_mxfp4_cutlass(nvfp4, mw, stream);
                    if (mw.data) {
                        wcache_.cutlass_mxfp4[ptr] = mw;
                        mx_total += mw.sf_bytes;
                        mx_count++;
                    }
                }
                if (mx_count > 0) {
                    cudaStreamSynchronize(stream);
                    wcache_.cutlass_mxfp4_bytes = mx_total;
                    IMP_LOG_INFO("CUTLASS sm_120 MXFP4 weight cache: %d tensors, %.2f MiB",
                                 mx_count, mx_total / (1024.0 * 1024.0));
                }
            }
        }

        // Native MXFP4 GGUF: unpack and register directly in CUTLASS cache.
        // Runs unconditionally (not inside the NVFP4 block).
        if (qscratch_.mxfp4_act_sf != nullptr && cutlass_sm120_mxfp4_available()) {
            int mx_native = 0;
            size_t mx_native_bytes = 0;
            auto register_if_mxfp4 = [&](const Tensor& w, GGMLQuantType qt) {
                if (qt != GGMLQuantType::MXFP4 || !w.data || !w.on_device) return;
                if (w.ndim < 2 || w.shape[1] % 32 != 0) return;
                if (wcache_.cutlass_mxfp4.count(w.data)) return;  // already registered
                CutlassMxFP4Weight mw;
                if (unpack_mxfp4_gguf(w.data, w.shape[0], w.shape[1], mw, stream)) {
                    wcache_.cutlass_mxfp4[w.data] = mw;
                    mx_native_bytes += mw.sf_bytes + static_cast<size_t>(w.shape[0]) * (w.shape[1] / 2);
                    mx_native++;
                }
            };
            for (int i = 0; i < cfg.n_layers; i++) {
                const auto& L = model_->layer(i);
                register_if_mxfp4(L.wq, L.wq_qtype);
                register_if_mxfp4(L.wk, L.wk_qtype);
                register_if_mxfp4(L.wv, L.wv_qtype);
                register_if_mxfp4(L.wo, L.wo_qtype);
                register_if_mxfp4(L.w_up, L.w_up_qtype);
                register_if_mxfp4(L.w_gate, L.w_gate_qtype);
                register_if_mxfp4(L.w_down, L.w_down_qtype);
            }
            register_if_mxfp4(model_->output_proj(), model_->out_proj_qtype_);
            if (mx_native > 0) {
                cudaStreamSynchronize(stream);
                wcache_.cutlass_mxfp4_bytes += mx_native_bytes;
                wcache_.use_mxfp4 = true;
                IMP_LOG_INFO("Native MXFP4 GGUF: %d tensors, %.2f MiB (direct → CUTLASS)",
                             mx_native, mx_native_bytes / (1024.0 * 1024.0));

                // Sync and check for errors from unpack kernels
                cudaStreamSynchronize(stream);
                { cudaError_t e = cudaGetLastError();
                  if (e != cudaSuccess) IMP_LOG_ERROR("MXFP4 unpack error: %s", cudaGetErrorString(e)); }

                // Dequant MXFP4 → FP16 for decode (GEMV needs FP16)
                size_t fp16_total = 0;
                for (auto& [ptr, mw] : wcache_.cutlass_mxfp4) {
                    if (wcache_.fp16.count(ptr)) continue;
                    size_t fp16_bytes = static_cast<size_t>(mw.N) * mw.K * sizeof(half);
                    void* d_fp16 = nullptr;
                    cudaError_t err = cudaMalloc(&d_fp16, fp16_bytes);
                    if (err != cudaSuccess) {
                        IMP_LOG_WARN("MXFP4 FP16 dequant alloc failed: %s (%.1f MiB)",
                                     cudaGetErrorString(err), fp16_bytes / (1024.0 * 1024.0));
                        continue;
                    }
                    dequant_mxfp4_to_fp16(mw, d_fp16, stream);
                    cudaStreamSynchronize(stream);
                    { cudaError_t e2 = cudaGetLastError();
                      if (e2 != cudaSuccess) {
                          IMP_LOG_ERROR("MXFP4 dequant kernel failed: %s [N=%lld K=%lld]",
                                       cudaGetErrorString(e2), (long long)mw.N, (long long)mw.K);
                          cudaFree(d_fp16);
                          continue;
                      }
                    }
                    int64_t shape[2] = {mw.N, mw.K};
                    wcache_.fp16[ptr] = Tensor(d_fp16, DType::FP16, 2, shape, true);
                    fp16_total += fp16_bytes;
                }
                if (fp16_total > 0) {
                    cudaStreamSynchronize(stream);
                    { cudaError_t e = cudaGetLastError();
                      if (e != cudaSuccess) IMP_LOG_ERROR("MXFP4 dequant kernel error: %s", cudaGetErrorString(e)); }
                    IMP_LOG_INFO("MXFP4 decode fallback: dequant → FP16 cache %.2f MiB",
                                 fp16_total / (1024.0 * 1024.0));
                }
            }
        }

        // Cache MoE expert weights — done after FP16 free so mode 2 has full budget
        int nvfp4_moe_count = 0;
        size_t nvfp4_moe_total = 0;
        size_t moe_budget;
        if (wcache_.nvfp4_decode_mode == 2) {
            size_t free_mem = 0, total_mem = 0;
            cudaMemGetInfo(&free_mem, &total_mem);
            constexpr size_t kMoeReserve = 128ULL * 1024 * 1024;
            moe_budget = (free_mem > kMoeReserve) ? (free_mem - kMoeReserve) : 0;
        } else {
            moe_budget = (remaining_budget > wcache_.nvfp4_bytes)
                         ? (remaining_budget - wcache_.nvfp4_bytes) : 0;
        }
        bool moe_budget_exhausted = false;

        auto cache_moe_expert_nvfp4 = [&](const Tensor& packed, GGMLQuantType qtype) {
            if (!packed.data) return;
            if (!nvfp4_beneficial(qtype)) return;
            if (wcache_.nvfp4_moe.count(packed.data)) return;
            if (moe_budget_exhausted) return;
            if (!packed.on_device) return;
            if (packed.ndim < 3) return;

            int ne = static_cast<int>(packed.shape[0]);
            int rows = static_cast<int>(packed.shape[1]);
            int cols = static_cast<int>(packed.shape[2]);
            if (cols % 16 != 0) return;
            if (!dequant_gpu_supported(qtype) || !qscratch_.dequant) return;

            size_t nvfp4_bytes = static_cast<size_t>(ne) * rows * cols / 2 +
                                 static_cast<size_t>(ne) * rows * cols / 16 +
                                 static_cast<size_t>(ne) * sizeof(float);

            if (nvfp4_moe_total + nvfp4_bytes > moe_budget) {
                moe_budget_exhausted = true;
                IMP_LOG_INFO("NVFP4 MoE cache: VRAM budget reached after %d MoE tensors "
                             "(%.1f / %.1f MiB)", nvfp4_moe_count,
                             nvfp4_moe_total / (1024.0 * 1024.0),
                             moe_budget / (1024.0 * 1024.0));
                return;
            }

            NvFP4MoEQuantResult result;
            quantize_packed_experts_to_nvfp4(
                packed.data, qtype, ne, rows, cols,
                qscratch_.dequant, result, stream);

            wcache_.nvfp4_moe[packed.data] = result;
            nvfp4_moe_total += nvfp4_bytes;
            nvfp4_moe_count++;
        };

        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            cache_moe_expert_nvfp4(L.expert_gate_packed, L.expert_gate_qtype);
            cache_moe_expert_nvfp4(L.expert_up_packed,   L.expert_up_qtype);
            cache_moe_expert_nvfp4(L.expert_down_packed,  L.expert_down_qtype);
        }

        if (nvfp4_moe_count > 0) {
            wcache_.nvfp4_moe_bytes = nvfp4_moe_total;
            IMP_LOG_INFO("NVFP4 MoE cache: %d tensors, %.2f MiB",
                         nvfp4_moe_count, nvfp4_moe_total / (1024.0 * 1024.0));
        } else if (wcache_.nvfp4.empty()) {
            IMP_LOG_INFO("NVFP4 decode: no eligible weights found (all ≤ 4.5 bits/elem)");
        }
    }
}

// ---------------------------------------------------------------------------
// Shared workspace configuration (pure pointer arithmetic, no allocation)
// ---------------------------------------------------------------------------

void GraphExecutor::configure_attn_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d   = cfg.d_model;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);


    char* ptr = static_cast<char*>(shared_workspace_);

    q_        = make_workspace_tensor(ptr, compute_dtype_, max_tokens, nh * hd,
                                      align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    // K and V are contiguous (no alignment gap) to enable strided batched GEMM.
    // v_.data == k_.data + kv_raw exactly, so output_stride = kv_raw / es.
    {
        size_t kv_raw = static_cast<size_t>(max_tokens) * nkv * hd * es;
        int64_t kv_shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(nkv * hd)};
        k_ = Tensor(ptr, compute_dtype_, 2, kv_shape, true);
        v_ = Tensor(ptr + kv_raw, compute_dtype_, 2, kv_shape, true);
        ptr += align256(2 * kv_raw);
    }
    attn_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, nh * hd,
                                      align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    proj_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d,
                                      align256(static_cast<size_t>(max_tokens) * d * es));
}

void GraphExecutor::configure_ffn_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d  = cfg.d_model;
    int ff = cfg.d_ff;
    size_t es = dtype_size(compute_dtype_);

    char* ptr = static_cast<char*>(shared_workspace_);

    gate_out_   = make_workspace_tensor(ptr, compute_dtype_, max_tokens, ff,
                                        align256(static_cast<size_t>(max_tokens) * ff * es));
    up_out_     = make_workspace_tensor(ptr, compute_dtype_, max_tokens, ff,
                                        align256(static_cast<size_t>(max_tokens) * ff * es));
    swiglu_out_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, ff,
                                        align256(static_cast<size_t>(max_tokens) * ff * es));
    ffn_out_    = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d,
                                        align256(static_cast<size_t>(max_tokens) * d * es));
}

void GraphExecutor::configure_moe_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d     = cfg.d_model;
    int ne    = cfg.n_experts;
    int top_k = cfg.n_experts_active;
    int eff   = max_expert_eff_;
    size_t es = dtype_size(compute_dtype_);
    int expanded = max_tokens * top_k;


    char* ptr = static_cast<char*>(shared_workspace_);

    // gate_logits: FP32
    moe_.gate_logits = make_workspace_tensor(ptr, DType::FP32, max_tokens, ne,
                                             align256(static_cast<size_t>(max_tokens) * ne * sizeof(float)));

    moe_.gathered      = make_workspace_tensor(ptr, compute_dtype_, expanded, d,
                                               align256(static_cast<size_t>(expanded) * d * es));
    moe_.expert_gate   = make_workspace_tensor(ptr, compute_dtype_, expanded, eff,
                                               align256(static_cast<size_t>(expanded) * eff * es));
    moe_.expert_up     = make_workspace_tensor(ptr, compute_dtype_, expanded, eff,
                                               align256(static_cast<size_t>(expanded) * eff * es));
    moe_.expert_swiglu = make_workspace_tensor(ptr, compute_dtype_, expanded, eff,
                                               align256(static_cast<size_t>(expanded) * eff * es));
    moe_.expert_down   = make_workspace_tensor(ptr, compute_dtype_, expanded, d,
                                               align256(static_cast<size_t>(expanded) * d * es));
    moe_.scatter_out   = make_workspace_tensor(ptr, DType::FP32, max_tokens, d,
                                               align256(static_cast<size_t>(max_tokens) * d * sizeof(float)));
}

void GraphExecutor::configure_ssm_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int inner = cfg.ssm_inner_size;
    int n_groups = cfg.ssm_group_count;
    int state_size = cfg.ssm_state_size;
    int n_heads = cfg.ssm_dt_rank;
    int conv_channels = inner + 2 * n_groups * state_size;
    int ssm_in_dim = inner + conv_channels + n_heads;
    size_t es = dtype_size(compute_dtype_);


    char* ptr = static_cast<char*>(shared_workspace_);

    // GDN layers need FP32 intermediate (4 bytes/elem) for numerical precision.
    // Non-GDN SSM layers only need FP16 (es bytes/elem).
    size_t proj_elem_size = has_gdn_ ? sizeof(float) : es;
    ssm_proj_buf_ = make_workspace_tensor(ptr, compute_dtype_, max_tokens, ssm_in_dim,
                                          align256(static_cast<size_t>(max_tokens) * ssm_in_dim * proj_elem_size));
    ssm_xBC_buf_  = make_workspace_tensor(ptr, compute_dtype_, max_tokens, conv_channels,
                                          align256(static_cast<size_t>(max_tokens) * conv_channels * es));
    ssm_y_buf_    = make_workspace_tensor(ptr, compute_dtype_, max_tokens, inner,
                                          align256(static_cast<size_t>(max_tokens) * inner * es));
    ssm_z_buf_    = make_workspace_tensor(ptr, compute_dtype_, max_tokens, inner,
                                          align256(static_cast<size_t>(max_tokens) * inner * es));
    ssm_out_buf_  = make_workspace_tensor(ptr, compute_dtype_, max_tokens, d,
                                          align256(static_cast<size_t>(max_tokens) * d * es));
    // GDN layers store BOTH alpha and beta projections in ssm_dt_buf_ (sequentially).
    // Allocate 2x n_heads to fit both. Non-GDN SSM only uses 1x (dt projection).
    size_t dt_multiplier = has_gdn_ ? 2 : 1;
    ssm_dt_buf_   = make_workspace_tensor(ptr, compute_dtype_, max_tokens, n_heads * dt_multiplier,
                                          align256(static_cast<size_t>(max_tokens) * n_heads * dt_multiplier * es));
}

bool GraphExecutor::resize_workspace(int new_max_tokens, cudaStream_t stream) {
    if (new_max_tokens == shared_workspace_max_tokens_ || new_max_tokens <= 0) return true;
    if (new_max_tokens > max_tokens_) new_max_tokens = max_tokens_;  // never exceed init-time max

    // Recompute shared sizes for the new token count
    int saved_max = max_tokens_;
    max_tokens_ = new_max_tokens;
    compute_shared_sizes(new_max_tokens);
    max_tokens_ = saved_max;

    size_t new_shared = std::max({attn_shared_size_, ffn_shared_size_,
                                  moe_shared_size_, ssm_shared_size_});
    if (new_shared == 0) return true;

    if (new_shared > shared_workspace_size_) {
        // Only reallocate when growing — reuse existing buffer if large enough.
        // This avoids expensive cudaMallocAsync/cudaFreeAsync on every batch size change.
        if (shared_workspace_) {
            cudaFreeAsync(shared_workspace_, stream);
        }
        cudaError_t err = cudaMallocAsync(&shared_workspace_, new_shared, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to resize shared workspace to %zu bytes: %s",
                          new_shared, cudaGetErrorString(err));
            shared_workspace_ = nullptr;
            shared_workspace_size_ = 0;
            return false;
        }
        shared_workspace_size_ = new_shared;
    }
    shared_workspace_max_tokens_ = new_max_tokens;
    return true;
}

// ---------------------------------------------------------------------------
// Dual workspace for concurrent prefill/decode overlap
// ---------------------------------------------------------------------------

bool GraphExecutor::allocate_decode_workspace(cudaStream_t stream, int max_batch) {
    if (decode_workspace_) return true;  // already allocated
    if (max_batch <= 0) max_batch = 1;

    const auto& cfg = model_->config();
    int dm = cfg.d_model;
    decode_max_batch_ = max_batch;

    // Persistent workspace for max_batch tokens: hidden + residual + norm_out + logits
    size_t persistent = static_cast<size_t>(dm) * sizeof(half) * 3 * max_batch  // hidden + residual + norm_out
                      + static_cast<size_t>(cfg.vocab_size) * sizeof(float) * max_batch;  // logits
    if (fp32_accum_buf_) persistent += static_cast<size_t>(dm) * sizeof(float) * max_batch;  // fp32_hidden

    decode_workspace_ = vram_alloc(vram_alloc_, persistent, "decode_workspace");
    if (!decode_workspace_) {
        IMP_LOG_ERROR("Failed to allocate decode workspace");
        return false;
    }
    decode_persistent_size_ = persistent;

    // Shared workspace for max_batch tokens
    int saved = max_tokens_;
    max_tokens_ = max_batch;
    compute_shared_sizes(max_batch);
    max_tokens_ = saved;

    size_t shared = std::max({attn_shared_size_, ffn_shared_size_,
                              moe_shared_size_, ssm_shared_size_});
    if (shared > 0) {
        decode_shared_workspace_ = vram_alloc(vram_alloc_, shared, "decode_shared_workspace");
        if (!decode_shared_workspace_) {
            IMP_LOG_ERROR("Failed to allocate decode shared workspace");
            vram_free(vram_alloc_, decode_workspace_);
            decode_workspace_ = nullptr;
            return false;
        }
    }
    decode_shared_size_ = shared;

    // Recompute sizes for original max_tokens
    compute_shared_sizes(max_tokens_);

    IMP_LOG_INFO("Decode overlap workspace: %.2f MiB for max_batch=%d (persistent=%.1f KB, shared=%.1f KB)",
                 (persistent + shared) / (1024.0 * 1024.0), max_batch,
                 persistent / 1024.0, shared / 1024.0);
    return true;
}

void GraphExecutor::use_workspace(int slot) {
    if (slot == active_workspace_) return;

    const auto& cfg = model_->config();
    int dm = cfg.d_model;

    if (slot == 1 && decode_workspace_) {
        // Save prefill workspace
        saved_prefill_ws_.persistent = persistent_workspace_;
        saved_prefill_ws_.persistent_size = persistent_workspace_size_;
        saved_prefill_ws_.shared = shared_workspace_;
        saved_prefill_ws_.shared_size = shared_workspace_size_;
        saved_prefill_ws_.shared_max_tokens = shared_workspace_max_tokens_;
        saved_prefill_ws_.hidden = hidden_;
        saved_prefill_ws_.residual = residual_;
        saved_prefill_ws_.norm_out = norm_out_;
        saved_prefill_ws_.logits = logits_;
        saved_prefill_ws_.fp32_accum = fp32_accum_buf_;
        saved_prefill_ws_.fp32_hidden = fp32_hidden_;

        // Switch to decode workspace
        persistent_workspace_ = decode_workspace_;
        persistent_workspace_size_ = decode_persistent_size_;
        shared_workspace_ = decode_shared_workspace_;
        shared_workspace_size_ = decode_shared_size_;
        shared_workspace_max_tokens_ = decode_max_batch_;

        // Set up tensor views into decode workspace (sized for decode_max_batch_)
        int mb = decode_max_batch_;
        char* p = static_cast<char*>(decode_workspace_);
        int64_t shape_mb[2] = {mb, dm};
        hidden_ = Tensor(p, DType::FP16, 2, shape_mb, true);
        p += static_cast<size_t>(dm) * sizeof(half) * mb;
        residual_ = Tensor(p, DType::FP16, 2, shape_mb, true);
        p += static_cast<size_t>(dm) * sizeof(half) * mb;
        norm_out_ = Tensor(p, DType::FP16, 2, shape_mb, true);
        p += static_cast<size_t>(dm) * sizeof(half) * mb;
        int64_t shape_logits[2] = {mb, cfg.vocab_size};
        logits_ = Tensor(p, DType::FP32, 2, shape_logits, true);
        p += static_cast<size_t>(cfg.vocab_size) * sizeof(float) * mb;
        if (saved_prefill_ws_.fp32_accum) {
            fp32_accum_buf_ = p;
            int64_t shape_fp32[2] = {mb, dm};
            fp32_hidden_ = Tensor(p, DType::FP32, 2, shape_fp32, true);
        }

        active_workspace_ = 1;
    } else if (slot == 0) {
        // Restore prefill workspace
        persistent_workspace_ = saved_prefill_ws_.persistent;
        persistent_workspace_size_ = saved_prefill_ws_.persistent_size;
        shared_workspace_ = saved_prefill_ws_.shared;
        shared_workspace_size_ = saved_prefill_ws_.shared_size;
        shared_workspace_max_tokens_ = saved_prefill_ws_.shared_max_tokens;
        hidden_ = saved_prefill_ws_.hidden;
        residual_ = saved_prefill_ws_.residual;
        norm_out_ = saved_prefill_ws_.norm_out;
        logits_ = saved_prefill_ws_.logits;
        fp32_accum_buf_ = saved_prefill_ws_.fp32_accum;
        fp32_hidden_ = saved_prefill_ws_.fp32_hidden;
        active_workspace_ = 0;
    }
}

bool GraphExecutor::layer_has_attention(int layer) const {
    return model_->layer(layer).wq.data != nullptr;
}

bool GraphExecutor::layer_has_ssm(int layer) const {
    return model_->layer(layer).ssm_in.data != nullptr;
}

bool GraphExecutor::layer_has_gdn(int layer) const {
    return model_->layer(layer).gdn_gate.data != nullptr;
}

bool GraphExecutor::layer_has_moe(int layer) const {
    const auto& ly = model_->layer(layer);
    return ly.moe_gate.data != nullptr;
}

bool GraphExecutor::layer_has_dense_ffn(int layer) const {
    const auto& ly = model_->layer(layer);
    return ly.w_up.data != nullptr && ly.moe_gate.data == nullptr;
}

Tensor GraphExecutor::view_tokens(const Tensor& buf, int n_tokens) const {
    // buf is always [max_tokens_, cols] from allocate_buffers.
    // Return a [n_tokens, cols] view.
    return slice_rows(buf, n_tokens);
}


void GraphExecutor::ensure_logits_pinned(int total_floats) {
    if (h_logits_pinned_ && h_logits_pinned_size_ >= total_floats) return;
    if (h_logits_pinned_) cudaFreeHost(h_logits_pinned_);
    cudaHostAlloc(&h_logits_pinned_, total_floats * sizeof(float), cudaHostAllocDefault);
    h_logits_pinned_size_ = total_floats;
}

} // namespace imp
